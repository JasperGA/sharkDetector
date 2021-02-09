#ifndef SSIM_HPP
#define SSIM_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/cuda.hpp>

#include <opencv2/cudaarithm/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <iostream>
#include <sstream> 

using namespace std;
using namespace cv;


Scalar getMSSIM_CUDA_optimized( const Mat& i1, const Mat& i2, BufferMSSIM& b)
{
    const float C1 = 6.5025f, C2 = 58.5225f;
    /***************************** INITS **********************************/
    b.gI1.upload(i1);
    b.gI2.upload(i2);
    gpu::Stream stream;
    b.gI1.convertTo(b.t1, CV_32F, stream);
    b.gI2.convertTo(b.t2, CV_32F, stream);
    gpu::split(b.t1, b.vI1, stream);
    gpu::split(b.t2, b.vI2, stream);
    Scalar mssim;
    Ptr<gpu::Filter> gauss = gpu::createGaussianFilter(b.vI1[0].type(), -1, Size(11, 11), 1.5);
    for( int i = 0; i < b.gI1.channels(); ++i )
    {
        cuda::multiply(b.vI2[i], b.vI2[i], b.I2_2, 1, -1, stream);        // I2^2
        cuda::multiply(b.vI1[i], b.vI1[i], b.I1_2, 1, -1, stream);        // I1^2
        cuda::multiply(b.vI1[i], b.vI2[i], b.I1_I2, 1, -1, stream);       // I1 * I2
        gauss->apply(b.vI1[i], b.mu1, stream);
        gauss->apply(b.vI2[i], b.mu2, stream);
        cuda::multiply(b.mu1, b.mu1, b.mu1_2, 1, -1, stream);
        cuda::multiply(b.mu2, b.mu2, b.mu2_2, 1, -1, stream);
        cuda::multiply(b.mu1, b.mu2, b.mu1_mu2, 1, -1, stream);
        gauss->apply(b.I1_2, b.sigma1_2, stream);
        cuda::subtract(b.sigma1_2, b.mu1_2, b.sigma1_2, cuda::GpuMat(), -1, stream);
        //b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation
        gauss->apply(b.I2_2, b.sigma2_2, stream);
        cuda::subtract(b.sigma2_2, b.mu2_2, b.sigma2_2, cuda::GpuMat(), -1, stream);
        //b.sigma2_2 -= b.mu2_2;
        gauss->apply(b.I1_I2, b.sigma12, stream);
        cuda::subtract(b.sigma12, b.mu1_mu2, b.sigma12, cuda::GpuMat(), -1, stream);
        //b.sigma12 -= b.mu1_mu2;
        //here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
        cuda::multiply(b.mu1_mu2, 2, b.t1, 1, -1, stream); //b.t1 = 2 * b.mu1_mu2 + C1;
        cuda::add(b.t1, C1, b.t1, cuda::GpuMat(), -1, stream);
        cuda::multiply(b.sigma12, 2, b.t2, 1, -1, stream); //b.t2 = 2 * b.sigma12 + C2;
        cuda::add(b.t2, C2, b.t2, cuda::GpuMat(), -12, stream);
        cuda::multiply(b.t1, b.t2, b.t3, 1, -1, stream);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
        cuda::add(b.mu1_2, b.mu2_2, b.t1, cuda::GpuMat(), -1, stream);
        cuda::add(b.t1, C1, b.t1, cuda::GpuMat(), -1, stream);
        cuda::add(b.sigma1_2, b.sigma2_2, b.t2, cuda::GpuMat(), -1, stream);
        cuda::add(b.t2, C2, b.t2, cuda::GpuMat(), -1, stream);
        cuda::multiply(b.t1, b.t2, b.t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        cuda::divide(b.t3, b.t1, b.ssim_map, 1, -1, stream);      // ssim_map =  t3./t1;
        stream.waitForCompletion();
        Scalar s = cuda::sum(b.ssim_map, b.buf);
        mssim.val[i] = s.val[0] / (b.ssim_map.rows * b.ssim_map.cols);
    }
    return mssim;
}

#endif