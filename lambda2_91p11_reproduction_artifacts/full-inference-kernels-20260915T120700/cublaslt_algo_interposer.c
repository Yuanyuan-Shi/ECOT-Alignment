#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cublasLt.h>

typedef cublasStatus_t (*matmul_fn_t)(
    cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*, cublasLtMatrixLayout_t,
    const void*, cublasLtMatrixLayout_t, const void*, const void*, cublasLtMatrixLayout_t,
    void*, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*, void*, size_t, cudaStream_t);
typedef cublasStatus_t (*get_attr_fn_t)(
    const cublasLtMatmulAlgo_t*, cublasLtMatmulAlgoConfigAttributes_t, void*, size_t, size_t*);

static matmul_fn_t real_matmul = NULL;
static get_attr_fn_t real_get_attr = NULL;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;

static void resolve_symbols(void) {
    real_matmul = (matmul_fn_t)dlsym(RTLD_NEXT, "cublasLtMatmul");
    real_get_attr = (get_attr_fn_t)dlsym(RTLD_NEXT, "cublasLtMatmulAlgoConfigGetAttribute");
}

static int get_i32(const cublasLtMatmulAlgo_t* algo, int attr, int32_t* value) {
    size_t written = 0;
    return real_get_attr(algo, (cublasLtMatmulAlgoConfigAttributes_t)attr, value, sizeof(*value), &written);
}

static int get_u32(const cublasLtMatmulAlgo_t* algo, int attr, uint32_t* value) {
    size_t written = 0;
    return real_get_attr(algo, (cublasLtMatmulAlgoConfigAttributes_t)attr, value, sizeof(*value), &written);
}

cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t compute_desc, const void* alpha,
    const void* A, cublasLtMatrixLayout_t A_desc, const void* B, cublasLtMatrixLayout_t B_desc,
    const void* beta, const void* C, cublasLtMatrixLayout_t C_desc, void* D,
    cublasLtMatrixLayout_t D_desc, const cublasLtMatmulAlgo_t* algo, void* workspace,
    size_t workspace_size, cudaStream_t stream) {
    pthread_once(&resolve_once, resolve_symbols);
    if (!real_matmul) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    const char* output_path = getenv("ECOT_CUBLASLT_CAPTURE_FILE");
    if (output_path && algo && real_get_attr) {
        int32_t algo_id = -1, split_k = -1;
        uint32_t tile = 0, reduction = 0, swizzle = 0, custom = 0, stages = 0;
        int s_id = get_i32(algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id);
        int s_tile = get_u32(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile);
        int s_split = get_i32(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k);
        int s_reduction = get_u32(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction);
        int s_swizzle = get_u32(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle);
        int s_custom = get_u32(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom);
        int s_stages = get_u32(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages);

        pthread_mutex_lock(&output_mutex);
        FILE* output = fopen(output_path, "a");
        if (output) {
            fprintf(output,
                    "{\"algo_id\":%d,\"tile_id\":%u,\"stages_id\":%u,\"split_k\":%d,"
                    "\"reduction_scheme\":%u,\"cta_swizzle\":%u,\"custom_option\":%u,"
                    "\"workspace_size_bytes\":%zu,\"workspace_is_null\":%s,"
                    "\"attribute_status\":{\"id\":%d,\"tile\":%d,\"stages\":%d,"
                    "\"split_k\":%d,\"reduction\":%d,\"swizzle\":%d,\"custom\":%d},"
                    "\"serialized_algo_u64_hex\":[",
                    algo_id, tile, stages, split_k, reduction, swizzle, custom, workspace_size,
                    workspace ? "false" : "true", s_id, s_tile, s_stages, s_split, s_reduction,
                    s_swizzle, s_custom);
            for (int index = 0; index < 8; ++index) {
                fprintf(output, "%s\"%016llx\"", index ? "," : "",
                        (unsigned long long)algo->data[index]);
            }
            fprintf(output, "]}\n");
            fclose(output);
        }
        pthread_mutex_unlock(&output_mutex);
    }

    return real_matmul(handle, compute_desc, alpha, A, A_desc, B, B_desc, beta, C, C_desc,
                       D, D_desc, algo, workspace, workspace_size, stream);
}
