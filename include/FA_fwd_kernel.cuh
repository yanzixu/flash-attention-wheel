#ifndef __FA_FWD_KERNEL__
#define __FA_FWD_KERNEL__

#include <math.h>

#define M_LOG2E 1.44269504088896340736 // log2(e)

namespace FA_FWD
{
    using namespace cute;
    struct Param
    {
        void *d_Q;
        void *d_K;
        void *d_V;
        void *d_O;

        int seq_len;

        int seq_stride;
        int qkv_seq_stride;
        int token_stride;
        int qkv_token_stride;
        int head_stride;
    };

    template <typename Param>
    __device__ __inline__ int get_global_offset(const int &batchId, const int &headId, const int &blockId, const Param &param)
    {
        return batchId * param.seq_stride + headId * param.head_stride + blockId * param.token_stride;
    }

    template <typename Param>
    __device__ __inline__ int get_qvk_global_offset(const int &batchId, const int &headId, const int &blockId, const Param &param)
    {
        return batchId * param.qkv_seq_stride + headId * param.head_stride + blockId * param.qkv_token_stride;
    }

    template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __inline__ void set_val(const Tensor<Engine0, Layout0> &src, Tensor<Engine1, Layout1> &dst)
    {
        // using SRCTYPE = typename Engine0::valtype;
    }

    template <typename Tensor0, typename Tensor1>
    __device__ __inline__ void transfer_tensor_data(const Tensor0 &src, Tensor1 &dst)
    {
        // using SRCTYPE = typename Tensor0::value_type;
        using DSTTYPE = typename Tensor1::value_type;
        static_assert(decltype(size(src) == size(dst))::value, "shape mis match");
        // TODO:通用性优化，如果两个张量数据类型相同，直接赋值，不需要static_cast
        for (size_t i = 0; i < size(src); i++)
        {
            dst(i) = static_cast<DSTTYPE>(src(i));
        }
    }

    template <typename Trait>
    __global__ void kFA_Fwd(Param param)
    {
        // using namespace cute;
        // TODO: declare trait type
        using T = typename Trait::T;
        static constexpr int kTileM = Trait::kTileM;
        static constexpr int kTileN = Trait::kTileN;
        static constexpr int kHeadDim = Trait::kHeadDim;

        static constexpr float kSCALE = 0.08838f; // 1/(128^(1/2))
        static constexpr float kSCALE_LOG2 = kSCALE * float(M_LOG2E);

        // TODO: index redefine
        const int batchId = blockIdx.z;
        const int headId = blockIdx.y;
        const int blockId = blockIdx.x;
        const int Id = threadIdx.x;

        // TODO: global tensor construct
        T *d_Q = reinterpret_cast<T *>(param.d_Q) + get_qvk_global_offset(batchId, headId, blockId * kTileM, param);
        T *d_K = reinterpret_cast<T *>(param.d_K) + get_qvk_global_offset(batchId, headId, 0, param);
        T *d_V = reinterpret_cast<T *>(param.d_V) + get_qvk_global_offset(batchId, headId, 0, param);
        T *d_O = reinterpret_cast<T *>(param.d_O) + get_global_offset(batchId, headId, blockId * kTileM, param);
        auto gQ = make_tensor(make_gmem_ptr(d_Q), make_layout(make_shape(Int<kTileM>{}, Int<kHeadDim>{}), make_stride(param.qkv_token_stride, _1{})));
        auto gK = make_tensor(make_gmem_ptr(d_K), make_layout(make_shape(Int<kTileN>{}, Int<kHeadDim>{}), make_stride(param.qkv_token_stride, _1{})));
        auto gV = make_tensor(make_gmem_ptr(d_V), make_layout(make_shape(Int<kTileN>{}, Int<kHeadDim>{}), make_stride(param.qkv_token_stride, _1{})));
        auto gO = make_tensor(make_gmem_ptr(d_O), make_layout(make_shape(Int<kTileM>{}, Int<kHeadDim>{}), make_stride(param.token_stride, _1{})));

        // TODO: SMEM tensor construct
        __shared__ T SMEM[Trait::SMEM_SIZE_Q + Trait::SMEM_SIZE_K + Trait::SMEM_SIZE_V];
        T *sm_Q = SMEM;
        T *sm_K = sm_Q + Trait::SMEM_SIZE_Q;
        T *sm_V = sm_K + Trait::SMEM_SIZE_K;
        auto sQ = make_tensor(make_smem_ptr(sm_Q), typename Trait::SMEM_LAYOUT_Q{});
        auto sK = make_tensor(make_smem_ptr(sm_K), typename Trait::SMEM_LAYOUT_K{});
        auto sV = make_tensor(make_smem_ptr(sm_V), typename Trait::SMEM_LAYOUT_V{});
        auto sVT = make_tensor(make_smem_ptr(sm_V), typename Trait::SMEM_LAYOUT_VT{});

        // TODO: g2s copy construct
        auto g2s_copy = typename Trait::G2S_COPY{};
        auto thr_g2s_copy = g2s_copy.get_slice(Id);
        auto g2s_gQ = thr_g2s_copy.partition_S(gQ);
        auto g2s_sQ = thr_g2s_copy.partition_D(sQ);
        auto g2s_gK = thr_g2s_copy.partition_S(gK);
        auto g2s_sK = thr_g2s_copy.partition_D(sK);
        auto g2s_gV = thr_g2s_copy.partition_S(gV);
        auto g2s_sV = thr_g2s_copy.partition_D(sV);

        // auto g2s_copyV = typename Trait::G2S_COPY_V{};
        // auto thr_g2s_copyV = g2s_copyV.get_slice(Id);
        // auto g2s_gV = thr_g2s_copyV.partition_S(gV);
        // auto g2s_sV = thr_g2s_copyV.partition_D(sV);

        // TODO: tiled_mma_first construct
        auto tiled_mma_first = typename Trait::TILED_MMA_FIRST{};
        auto thr_tiled_mma_first = tiled_mma_first.get_slice(Id);
        auto rQ = thr_tiled_mma_first.partition_fragment_A(gQ);
        auto rK = thr_tiled_mma_first.partition_fragment_B(gK);
        // SHOW_TENSOR(gQ)

        // TODO: construct row_col layout of scores matrix
        auto rScores_MMA = thr_tiled_mma_first.partition_fragment_C(make_tensor(Layout<Shape<Int<kTileM>, Int<kTileN>>>{}));
        static_assert(is_rmem<decltype(rScores_MMA)>::value, "rScores_MMA not reg");
        auto div_first = logical_divide(rScores_MMA.layout(), Shape<_2>{});
        static_assert(is_layout<decltype(div_first)>::value, "div_first not layout");
        // SHOW(div_first)
        // 这里使用make_layout函数的嵌套，这里的get<1>(div_first) -> layout，因此构造出来的tensor stride和原来同步（虽然寄存器数组，因为本质是一个view，因此stride会影响实际计算结果）
        auto rScores_RC_View = make_tensor(rScores_MMA.data(), make_layout(make_layout(get<1>(get<0>(div_first)), get<1>(div_first)),
                                                                           make_layout(get<0>(get<0>(div_first)), get<2>(div_first)))); // 这里rP_fp32没打印出地址，但是实际上是一个view
        auto rMax = make_tensor<typename Trait::AccType>(Shape<Int<size<0>(rScores_RC_View)>>{});
        auto rSum = make_fragment_like(rMax);

        // TODO: tiled_mma_second construct
        auto tiled_mma_second = typename Trait::TILED_MMA_SECOND{};
        auto thr_tiled_mma_second = tiled_mma_second.get_slice(Id);
        auto rV = thr_tiled_mma_second.partition_fragment_B(make_tensor(Layout<Shape<Int<kHeadDim>, Int<kTileN>>>{}));
        auto tmpL = rV.layout();
        auto rV_RC_View = make_tensor(rV.data(), make_layout(make_layout(get<1>(tmpL)),
                                                             make_layout(get<0>(tmpL), get<2>(tmpL))));
        // SHOW(rV)
        // SHOW(rV_RC_View)
        // auto rO = thr_tiled_mma_second.partition_fragment_C(gO);
        auto rO = thr_tiled_mma_second.partition_fragment_C(make_tensor(Layout<Shape<Int<kTileM>, Int<kHeadDim>>>{}));
        auto div_2nd = logical_divide(rO.layout(), Shape<_2>{});
        auto rO_RC_View = make_tensor(rO.data(), make_layout(make_layout(get<1>(get<0>(div_2nd)), get<1>(div_2nd)),
                                                             make_layout(get<0>(get<0>(div_2nd)), get<2>(div_2nd)))); // 这里rP_fp32没打印出地址，但是实际上是一个view
        {
            // TODO:关于rV 切分不同的张量，传入gemm函数编译无法通过的问题
            //  rQ 通过partition_fragment_A sQ得到的结果可以进行gemm运算
            //  auto rQ_F = thr_tiled_mma_first.partition_fragment_A(sQ);
            //  SHOW(rQ_F)
            // partition_fragment_B(sVT)-> raw_ptr_16b(0000023FCFFFFBF0) o S<1,0,-2> o _0 o ((_2,_2),_16,_2):((_1,_2),_4,_64)
            // 暂时怀疑包含swizzle 成分是gemm函数传参后编译失败的原因
            // auto rV_F = thr_tiled_mma_second.partition_fragment_B(sVT);
            // SHOW(rV_F)
            // 并非 按照行方向swizzle就可以得到正确的寄存器张量
            // auto sVTest = make_tensor(make_smem_ptr(sm_V), typename Trait::SMEM_ATOM_Test{});
            // auto rV_Test = thr_tiled_mma_second.partition_fragment_B(sVTest);
            // SHOW(rV_Test)
        }

        // TODO: construct rP
        auto rP_RC = make_tensor<T>(rScores_RC_View.layout());
        auto div_3rd = logical_divide(rP_RC.layout(), Shape<Underscore, Shape<Underscore, _2>>{});
        auto rP_MMA_View = make_tensor(rP_RC.data(), make_layout(make_layout(get<0>(get<1>(div_3rd)), get<0>(get<0>(div_3rd)), get<0>(get<1>(get<1>(div_3rd)))),
                                                                 get<1>(get<0>(div_3rd)),
                                                                 get<1>(get<1>(get<1>(div_3rd)))));

        // TODO: s2r copy construct
        auto s2r_copy_Q = make_tiled_copy_A(typename Trait::S2R_ATOM{}, tiled_mma_first);
        auto thr_s2r_copy_Q = s2r_copy_Q.get_slice(Id);
        auto s2r_sQ = thr_s2r_copy_Q.partition_S(sQ);
        auto s2r_rQ = thr_s2r_copy_Q.retile_D(rQ);
        auto s2r_copy_K = make_tiled_copy_B(typename Trait::S2R_ATOM{}, tiled_mma_first);
        auto thr_s2r_copy_K = s2r_copy_K.get_slice(Id);
        auto s2r_sK = thr_s2r_copy_K.partition_S(sK);
        auto s2r_rK = thr_s2r_copy_K.retile_D(rK); // 这里的retile将N方向数据分加载进行压缩，因此K方向的迭代次数不受影响
        auto s2r_copy_V = make_tiled_copy_B(typename Trait::S2R_ATOM_V{}, tiled_mma_first);
        auto thr_s2r_copy_V = s2r_copy_V.get_slice(Id);
        auto s2r_sV = thr_s2r_copy_V.partition_S(sVT);
        auto s2r_rV = thr_s2r_copy_V.retile_D(rV);

        // TODO: sequence iteration
        constexpr int kFirstKCount = size<2>(s2r_rQ);
        constexpr int kSecondCount = size<2>(s2r_rV);
        // TODO: g2s copy
        // load Q illegal addr access -> gird & block position fused
        copy(g2s_copy, g2s_gQ, g2s_sQ);
        copy(g2s_copy, g2s_gK, g2s_sK);
        cp_async_fence();
        clear(rO);
        // error :地址加减操作是针对 thr_copy，即地址是私有数据，与张量地址无关
        for (int it_N = 0; it_N < param.seq_len; it_N += kTileN)
        {
            clear(rScores_MMA);

            cp_async_wait<0>();
            __syncthreads();

            // copy(g2s_copyV, g2s_gV, g2s_sV);
            copy(g2s_copy, g2s_gV, g2s_sV);
            g2s_gV.data() = g2s_gV.data() + kTileN * param.qkv_token_stride;
            cp_async_fence();

            // TODO: Q * Kt
            for (int it_K = 0; it_K < kFirstKCount; it_K++)
            {
                copy(s2r_copy_Q, s2r_sQ(_, _, it_K), s2r_rQ(_, _, it_K));
                copy(s2r_copy_K, s2r_sK(_, _, it_K), s2r_rK(_, _, it_K));
                gemm(tiled_mma_first, rScores_MMA, rQ(_, _, it_K), rK(_, _, it_K), rScores_MMA);
            }

            // TODO:debug scores val
            {
                // for (int m = 0; m < size<0>(rScores_RC_View); m++)
                // {
                //     for (int n = 0; n < size<1>(rScores_RC_View); n++)
                //     {
                //         rScores_RC_View(m, n) = rScores_RC_View(m, n) * kSCALE;
                //     }
                // }
                // SHOW_TENSOR(rScores_RC_View)
            }

            // TODO: online softmax
            if (it_N == 0)
            {
                // SHOW_TENSOR(sV)
                for (int m = 0; m < size<0>(rScores_RC_View); m++)
                {
                    float cur_max = -INFINITY;

                    for (int n = 0; n < size<1>(rScores_RC_View); n++)
                    {
                        cur_max = max(cur_max, rScores_RC_View(m, n));
                    }
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 0x01));
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 0x02));
                    rMax(m) = cur_max;
                    float local_sum = 0.f;
                    for (int n = 0; n < size<1>(rScores_RC_View); n++)
                    {
                        // rScores_RC_View(m, n) = exp(rScores_RC_View(m, n) - cur_max);
                        rScores_RC_View(m, n) = exp2f(kSCALE_LOG2 * (rScores_RC_View(m, n) - cur_max));
                        local_sum += rScores_RC_View(m, n);
                    }
                    rSum(m) = local_sum;
                }
            }
            else
            {
                for (int m = 0; m < size<0>(rScores_RC_View); m++)
                {
                    float cur_max = -INFINITY;
                    for (int n = 0; n < size<1>(rScores_RC_View); n++)
                    {
                        cur_max = max(cur_max, rScores_RC_View(m, n));
                    }
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 0x01));
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 0x02));
                    cur_max = max(cur_max, rMax(m));
                    float scale = exp2f(kSCALE_LOG2 * (rMax(m) - cur_max));
                    // 从it == 1 开始需要对rO更新的max因子
                    for (int n = 0; n < size<1>(rO_RC_View); n++)
                    {
                        rO_RC_View(m, n) = rO_RC_View(m, n) * scale;
                        // rO_RC_View(m, n) = rO_RC_View(m, n) * exp2f(float(M_LOG2E) * (rMax(m) - cur_max));
                        // rO_RC_View(m, n) = rO_RC_View(m, n) * exp((rMax(m) - cur_max));
                    }
                    rMax(m) = cur_max;
                    float local_sum = 0.f;
                    for (int n = 0; n < size<1>(rScores_RC_View); n++)
                    {
                        rScores_RC_View(m, n) = exp2f(kSCALE_LOG2 * (rScores_RC_View(m, n) - cur_max));
                        local_sum += rScores_RC_View(m, n);
                    }
                    rSum(m) = rSum(m) * scale + local_sum;
                }
            }

            // 这里处理rO需要单独保存上次迭代的max 和 当前的max值
            // for (int m = 0; m < size<0>(rO_RC_View); m++)
            // {
            //     for (int n = 0; n < size<1>(rO_RC_View); n++)
            //     {
            //         rO_RC_View(m, n) = rO_RC_View(m, n) * exp2f(float(M_LOG2E) * ())
            //     }
            // }

            transfer_tensor_data(rScores_RC_View, rP_RC);
            cp_async_wait<0>();
            __syncthreads();
            // TODO: matrix k g2s
            if (it_N < (param.seq_len - kTileN))
            {
                g2s_gK.data() = g2s_gK.data() + kTileN * param.qkv_token_stride;
                copy(g2s_copy, g2s_gK, g2s_sK);
                cp_async_fence();
            }

            // TODO: P * V
            for (int it_K = 0; it_K < kSecondCount; it_K++)
            {
                // 这里每次load两倍的数据，在N方向上拓展，因此it_K做rV在K维度上的迭代索引没问题
                copy(s2r_copy_V, s2r_sV(_, _, it_K), s2r_rV(_, _, it_K));
                gemm(tiled_mma_second, rO, rP_MMA_View(_, _, it_K), rV(_, _, it_K), rO);
            }
        }

        // TODO:epilogue r2s
        for (int m = 0; m < size<0>(rO_RC_View); m++)
        {
            rSum(m) = rSum(m) + __shfl_xor_sync(0xffffffff, rSum(m), 0x01);
            rSum(m) = rSum(m) + __shfl_xor_sync(0xffffffff, rSum(m), 0x02);
            float scale = 1.f / rSum(m);
            for (int n = 0; n < size<1>(rO_RC_View); n++)
            {
                rO_RC_View(m, n) = rO_RC_View(m, n) * scale;
            }
        }

        // TODO:debug
        {
            // g2s_gK.data() = g2s_gK.data() + (0 - 128) * param.qkv_token_stride;
            // g2s_gV.data() = g2s_gV.data() + (0 - 128) * param.qkv_token_stride;
            // clear(rO_RC_View);
            // for (int it_N = 0; it_N < param.seq_len; it_N += kTileN, g2s_gK.data() = g2s_gK.data() + kTileN * param.qkv_token_stride, g2s_gV.data() = g2s_gV.data() + kTileN * param.qkv_token_stride)
            // {
            //     clear(rScores_MMA);

            //     // TODO: g2s copy
            //     copy(g2s_copy, g2s_gQ, g2s_sQ); // load Q illegal addr access -> gird & block position fused
            //     copy(g2s_copy, g2s_gK, g2s_sK);
            //     cp_async_fence();
            //     cp_async_wait<0>();
            //     __syncthreads();
            //     copy(g2s_copy, g2s_gV, g2s_sV);
            //     cp_async_fence();

            //     // TODO: Q * Kt
            //     for (int it_K = 0; it_K < kFirstKCount; it_K++)
            //     {
            //         copy(s2r_copy_Q, s2r_sQ(_, _, it_K), s2r_rQ(_, _, it_K));
            //         copy(s2r_copy_K, s2r_sK(_, _, it_K), s2r_rK(_, _, it_K));
            //         gemm(tiled_mma_first, rScores_MMA, rQ(_, _, it_K), rK(_, _, it_K), rScores_MMA);
            //     }
            //     cp_async_wait<0>();
            //     __syncthreads();
            //     // for (int m = 0; m < size<0>(rScores_RC_View); m++)
            //     // {
            //     //     for (int n = 0; n < size<1>(rScores_RC_View); n++)
            //     //     {
            //     //         rScores_RC_View(m, n) = rScores_RC_View(m, n) * kSCALE;
            //     //     }
            //     // }
            //     // SHOW_TENSOR(rScores_RC_View)
            //     for (int m = 0; m < size<0>(rScores_RC_View); m++)
            //     {
            //         float scale = 1.f / rSum(m);
            //         for (int n = 0; n < size<1>(rScores_RC_View); n++)
            //         {
            //             rScores_RC_View(m, n) = exp2f(kSCALE_LOG2 * (rScores_RC_View(m, n) - rMax(m)));
            //             rScores_RC_View(m, n) = rScores_RC_View(m, n) * scale;
            //         }
            //     }
            //     // SHOW_TENSOR(rScores_RC_View)
            //     transfer_tensor_data(rScores_RC_View, rP_RC);

            //     // TODO: P * V
            //     for (int it_K = 0; it_K < kSecondCount; it_K++)
            //     {
            //         // 这里每次load两倍的数据，在N方向上拓展，因此it_K做rV在K维度上的迭代索引没问题
            //         copy(s2r_copy_V, s2r_sV(_, _, it_K), s2r_rV(_, _, it_K));
            //         gemm(tiled_mma_second, rO, rP_MMA_View(_, _, it_K), rV(_, _, it_K), rO);
            //     }
            //     SHOW_TENSOR(rO_RC_View)
            // }
        }
        // SHOW_TENSOR(rO_RC_View)
        // SHOW_TENSOR(rSum)

        auto rO_St = make_tensor<T>(rO.layout());
        transfer_tensor_data(rO, rO_St);
        // SHOW_TENSOR(rO_RC_View)
        // SHOW_TENSOR(rO_St)

        auto sO = make_tensor(sQ.data(), typename Trait::SMEM_LAYOUT_O{});
        auto r2s_copy_O = make_tiled_copy_C(typename Trait::R2S_ATOM{}, tiled_mma_second);
        auto thr_r2s_copy_O = r2s_copy_O.get_slice(Id);
        auto r2s_rO = thr_r2s_copy_O.retile_S(rO_St);
        auto r2s_sO = thr_r2s_copy_O.partition_D(sO);
        // TODO:test universal copy result tensor layout
        {
            // 这里不管使用128bit 还是 32bit的拷贝，切分出的张量结果都是一样的
            // tiled mma和copy atom是怎么工作的，为什么结果一致
            // auto r2s_copy_O_test = make_tiled_copy_C(typename Trait::R2S_ATOM_TEST{}, tiled_mma_second);
            // auto thr_r2s_copy_O_test = r2s_copy_O_test.get_slice(Id);
            // auto test_sO = thr_r2s_copy_O_test.partition_D(sO);
            // SHOW(r2s_sO)
            // SHOW(test_sO)
        }
        copy(r2s_copy_O, r2s_rO, r2s_sO);
        __syncthreads();
        // TODO: epilogue s2g
        typename Trait::S2G_COPY s2g_copy_O;
        auto thr_s2g_copy_O = s2g_copy_O.get_slice(Id);
        auto s2g_sO = thr_s2g_copy_O.partition_S(sO);
        auto s2g_gO = thr_s2g_copy_O.partition_D(gO);
        copy(s2g_copy_O, s2g_sO, s2g_gO);
    }

}

#endif // !__FA_FWD_KERNEL__
