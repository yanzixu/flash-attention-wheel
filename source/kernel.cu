#include <cute/tensor.hpp>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "utils/utils.h"
#include "include/FA_trait.cuh"
#include "include/FA_fwd_kernel.cuh"

using namespace cute;
using namespace std;

template <typename T>
void check(void *ref, void *data, int length)
{
    T *p_ref = reinterpret_cast<T *>(ref);
    T *p_data = reinterpret_cast<T *>(data);
    int count = 0;
    for (int i = 0; i < length; i++)
    {
        float locla_ref = static_cast<float>(p_ref[i]);
        float local_data = static_cast<float>(p_data[i]);
        if (fabs(locla_ref - local_data) > 0.0001f + 0.02f * fabs(locla_ref))
        {
            printf("%d ref %f data %f \n", i, locla_ref, local_data);
            count++;
            // return;
        }
    }
    printf("suc %d \n");
}

template <typename T>
void init(void *data, int row, int col)
{
    T *p_data = reinterpret_cast<T *>(data);
    for (int r = 0; r < row; r++)
    {
        for (int c = 0; c < 2 * col; c++)
        {
            p_data[r * 3 * col + c] = T(r % 8 + r / 8);
        }
        for (int c = 2 * col; c < 3 * col; c++)
        {
            p_data[r * 3 * col + c] = T(1);
        }
    }
}

void print_score(int length)
{
    float *h_score;
    h_score = (float *)(malloc(length * length * sizeof(float)));
    ifstream f;
    // f.open("C:\\code\\flash-attention\\scores_single.bin", ios::in | ios::binary);
    f.open("C:\\code\\flash-attention\\attention_single.bin", ios::in | ios::binary);
    if (!f.is_open())
    {
        printf("binary file open fail \n ");
        return;
    }
    f.read(reinterpret_cast<char *>(h_score), length * length * sizeof(float));
    f.close();
    for (size_t m = 0; m < length; m++)
    {
        for (size_t i = 0; i < length; i++)
        {
            printf("%f ", h_score[m * length + i]);
        }
        print("\n");
    }
}
void print_V(half *p, int row, int col, int stride)
{
    for (size_t r = 0; r < row; r++)
    {
        for (size_t i = 0; i < col; i++)
        {
            printf("%f ", float(p[r * stride + i]));
        }
        printf("\n");
    }
}

void test_FA_fwd()
{

    using T = half_t;
    constexpr int kBATCH = 1;
    constexpr int kSEQ_LEN = 2048;
    constexpr int kHEADS = 32;
    // constexpr int kHEADS = 1;
    constexpr int kHEADDIM = 128;
    constexpr int kQKV_SIZE = kBATCH * kSEQ_LEN * 3 * kHEADS * kHEADDIM * sizeof(T);
    constexpr int kO_SIZE = kBATCH * kSEQ_LEN * kHEADS * kHEADDIM * sizeof(T);
    // constexpr bool DEBUG = true;
    constexpr bool DEBUG = false;
    // TODO: init host memory
    void *h_qkv, *h_o, *h_o_ref;
    h_qkv = malloc(kQKV_SIZE);
    h_o = malloc(kO_SIZE);
    h_o_ref = malloc(kO_SIZE);
    ifstream f_qkv, f_o;
    f_qkv.open("C:\\code\\flash-attention\\qkv.bin", ios::in | ios::binary);
    f_o.open("C:\\code\\flash-attention\\out.bin", ios::in | ios::binary);
    // f_qkv.open("C:\\code\\flash-attention\\qkv_single.bin", ios::in | ios::binary);
    // f_o.open("C:\\code\\flash-attention\\out_single.bin", ios::in | ios::binary);
    if (!f_qkv.is_open() || !f_o.is_open())
    {
        printf("binary file open fail \n ");
        return;
    }
    f_qkv.read(reinterpret_cast<char *>(h_qkv), kQKV_SIZE);
    f_o.read(reinterpret_cast<char *>(h_o_ref), kO_SIZE);
    f_qkv.close();
    f_o.close();
    if (DEBUG)
    {
        init<T>(h_qkv, kSEQ_LEN, kHEADDIM);
    }
    // print_V((half *)(h_qkv) + 2 * kHEADDIM, kSEQ_LEN, kHEADDIM, 3 * kHEADDIM);
    // return;

    // TODO: init device memory
    void *d_qvk,
        *d_o;
    cudaMalloc(&d_qvk, kQKV_SIZE);
    cudaMalloc(&d_o, kO_SIZE);
    cudaMemcpy(d_qvk, h_qkv, kQKV_SIZE, cudaMemcpyHostToDevice);
    // TODO: layout BHSE,init param
    FA_FWD::Param param;
    param.head_stride = kHEADDIM;
    param.token_stride = kHEADS * param.head_stride;
    param.seq_stride = kSEQ_LEN * param.token_stride;
    param.qkv_token_stride = 3 * param.token_stride;
    param.qkv_seq_stride = kSEQ_LEN * param.qkv_token_stride;
    param.seq_len = kSEQ_LEN;
    param.d_Q = d_qvk;
    param.d_K = reinterpret_cast<T *>(d_qvk) + kHEADS * param.head_stride;
    param.d_V = reinterpret_cast<T *>(d_qvk) + 2 * kHEADS * param.head_stride;
    param.d_O = d_o;
    // TODO: construct launch config
    static constexpr int kTILE_M = 128;
    static constexpr int kTILE_N = 32;
    dim3 block(128, 1, 1);
    dim3 grid(kSEQ_LEN / kTILE_M, kHEADS, kBATCH);
    // FA_FWD_CONFIG::yan_trait<T, 128, 32, kHEADDIM, 4> trait;
    FA_FWD::kFA_Fwd<FA_FWD_CONFIG::yan_trait<T, kTILE_M, kTILE_N, kHEADDIM, 4>><<<grid, block>>>(param);
    cudaDeviceSynchronize();
    cudaError_t rtn = cudaGetLastError();
    if (rtn != cudaSuccess)
    {
        printf("err %s \n", cudaGetErrorString(rtn));
    }

    // TODO:copy data and check
    cudaMemcpy(h_o, d_o, kO_SIZE, cudaMemcpyDeviceToHost);
    if (!DEBUG)
    {
        check<T>(h_o_ref, h_o, kSEQ_LEN * kHEADS * kHEADDIM);
    }

    cudaFree(d_qvk);
    cudaFree(d_o);
    free(h_qkv);
    free(h_o);
    free(h_o_ref);
}

int main(int argc, char **argv)
{
    test_FA_fwd();
    return 0;
}