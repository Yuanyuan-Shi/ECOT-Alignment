#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cublasLt.h>
#include <cublas_v2.h>

static __thread char ecot_context[768] = "unassigned";
static pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long sequence_number = 0;

void ecot_set_context(const char* value) {
    if (!value) value = "unassigned";
    snprintf(ecot_context, sizeof(ecot_context), "%s", value);
}

static FILE* open_output(void) {
    const char* path = getenv("ECOT_CUBLAS_CAPTURE_FILE");
    return path ? fopen(path, "a") : NULL;
}

static unsigned long long next_sequence(void) {
    return __sync_fetch_and_add(&sequence_number, 1);
}

typedef cublasStatus_t (*lt_matmul_fn_t)(
    cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*, cublasLtMatrixLayout_t,
    const void*, cublasLtMatrixLayout_t, const void*, const void*, cublasLtMatrixLayout_t,
    void*, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*, void*, size_t, cudaStream_t);
typedef cublasStatus_t (*lt_algo_get_fn_t)(
    const cublasLtMatmulAlgo_t*, cublasLtMatmulAlgoConfigAttributes_t, void*, size_t, size_t*);
typedef cublasStatus_t (*lt_layout_get_fn_t)(
    cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, void*, size_t, size_t*);
typedef cublasStatus_t (*lt_desc_get_fn_t)(
    cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, void*, size_t, size_t*);

static lt_matmul_fn_t real_lt_matmul = NULL;
static lt_algo_get_fn_t real_lt_algo_get = NULL;
static lt_layout_get_fn_t real_lt_layout_get = NULL;
static lt_desc_get_fn_t real_lt_desc_get = NULL;
static pthread_once_t lt_once = PTHREAD_ONCE_INIT;

static void resolve_lt(void) {
    real_lt_matmul = (lt_matmul_fn_t)dlsym(RTLD_NEXT, "cublasLtMatmul");
    real_lt_algo_get = (lt_algo_get_fn_t)dlsym(RTLD_NEXT, "cublasLtMatmulAlgoConfigGetAttribute");
    real_lt_layout_get = (lt_layout_get_fn_t)dlsym(RTLD_NEXT, "cublasLtMatrixLayoutGetAttribute");
    real_lt_desc_get = (lt_desc_get_fn_t)dlsym(RTLD_NEXT, "cublasLtMatmulDescGetAttribute");
}

static int lt_algo_i32(const cublasLtMatmulAlgo_t* algo, int attr, int32_t* value) {
    size_t written = 0;
    return real_lt_algo_get(algo, (cublasLtMatmulAlgoConfigAttributes_t)attr, value, sizeof(*value), &written);
}

static int lt_algo_u32(const cublasLtMatmulAlgo_t* algo, int attr, uint32_t* value) {
    size_t written = 0;
    return real_lt_algo_get(algo, (cublasLtMatmulAlgoConfigAttributes_t)attr, value, sizeof(*value), &written);
}

static void lt_layout_values(cublasLtMatrixLayout_t layout, int32_t* type, uint64_t* rows,
                             uint64_t* cols, int64_t* ld) {
    size_t written = 0;
    *type = -1; *rows = 0; *cols = 0; *ld = 0;
    if (!real_lt_layout_get || !layout) return;
    real_lt_layout_get(layout, CUBLASLT_MATRIX_LAYOUT_TYPE, type, sizeof(*type), &written);
    real_lt_layout_get(layout, CUBLASLT_MATRIX_LAYOUT_ROWS, rows, sizeof(*rows), &written);
    real_lt_layout_get(layout, CUBLASLT_MATRIX_LAYOUT_COLS, cols, sizeof(*cols), &written);
    real_lt_layout_get(layout, CUBLASLT_MATRIX_LAYOUT_LD, ld, sizeof(*ld), &written);
}

cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t compute_desc, const void* alpha,
    const void* A, cublasLtMatrixLayout_t A_desc, const void* B, cublasLtMatrixLayout_t B_desc,
    const void* beta, const void* C, cublasLtMatrixLayout_t C_desc, void* D,
    cublasLtMatrixLayout_t D_desc, const cublasLtMatmulAlgo_t* algo, void* workspace,
    size_t workspace_size, cudaStream_t stream) {
    pthread_once(&lt_once, resolve_lt);
    if (!real_lt_matmul) return CUBLAS_STATUS_INTERNAL_ERROR;

    int32_t algo_id = -1, split_k = -1;
    uint32_t tile = 0, reduction = 0, swizzle = 0, custom = 0, stages = 0;
    int status[7] = {-1,-1,-1,-1,-1,-1,-1};
    if (algo && real_lt_algo_get) {
        status[0] = lt_algo_i32(algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id);
        status[1] = lt_algo_u32(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile);
        status[2] = lt_algo_u32(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages);
        status[3] = lt_algo_i32(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k);
        status[4] = lt_algo_u32(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction);
        status[5] = lt_algo_u32(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle);
        status[6] = lt_algo_u32(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom);
    }

    int32_t at, bt, ct, dt;
    uint64_t ar, ac, br, bc, cr, cc, dr, dc;
    int64_t ald, bld, cld, dld;
    lt_layout_values(A_desc, &at, &ar, &ac, &ald);
    lt_layout_values(B_desc, &bt, &br, &bc, &bld);
    lt_layout_values(C_desc, &ct, &cr, &cc, &cld);
    lt_layout_values(D_desc, &dt, &dr, &dc, &dld);

    int32_t compute_type = -1, scale_type = -1, trans_a = -1, trans_b = -1, epilogue = -1;
    if (real_lt_desc_get && compute_desc) {
        size_t written = 0;
        real_lt_desc_get(compute_desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, &compute_type, sizeof(compute_type), &written);
        real_lt_desc_get(compute_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type), &written);
        real_lt_desc_get(compute_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a), &written);
        real_lt_desc_get(compute_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b), &written);
        real_lt_desc_get(compute_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue), &written);
    }

    pthread_mutex_lock(&output_mutex);
    FILE* output = open_output();
    if (output) {
        fprintf(output,
            "{\"sequence\":%llu,\"api\":\"cublasLtMatmul\",\"context\":\"%s\","
            "\"algo_id\":%d,\"tile_id\":%u,\"stages_id\":%u,\"split_k\":%d,"
            "\"reduction_scheme\":%u,\"cta_swizzle\":%u,\"custom_option\":%u,"
            "\"workspace_size_bytes\":%zu,\"workspace_is_null\":%s,"
            "\"compute_type\":%d,\"scale_type\":%d,\"trans_a\":%d,\"trans_b\":%d,\"epilogue\":%d,"
            "\"A\":{\"type\":%d,\"rows\":%llu,\"cols\":%llu,\"ld\":%lld},"
            "\"B\":{\"type\":%d,\"rows\":%llu,\"cols\":%llu,\"ld\":%lld},"
            "\"C\":{\"type\":%d,\"rows\":%llu,\"cols\":%llu,\"ld\":%lld},"
            "\"D\":{\"type\":%d,\"rows\":%llu,\"cols\":%llu,\"ld\":%lld},"
            "\"attribute_status\":[%d,%d,%d,%d,%d,%d,%d],\"serialized_algo_u64_hex\":[",
            next_sequence(), ecot_context, algo_id, tile, stages, split_k, reduction, swizzle, custom,
            workspace_size, workspace ? "false" : "true", compute_type, scale_type, trans_a, trans_b, epilogue,
            at,(unsigned long long)ar,(unsigned long long)ac,(long long)ald,
            bt,(unsigned long long)br,(unsigned long long)bc,(long long)bld,
            ct,(unsigned long long)cr,(unsigned long long)cc,(long long)cld,
            dt,(unsigned long long)dr,(unsigned long long)dc,(long long)dld,
            status[0],status[1],status[2],status[3],status[4],status[5],status[6]);
        if (algo) {
            for (int i=0;i<8;++i) fprintf(output,"%s\"%016llx\"",i?",":"",(unsigned long long)algo->data[i]);
        }
        fprintf(output,"]}\n");
        fclose(output);
    }
    pthread_mutex_unlock(&output_mutex);
    return real_lt_matmul(handle, compute_desc, alpha, A, A_desc, B, B_desc, beta, C, C_desc,
                          D, D_desc, algo, workspace, workspace_size, stream);
}
