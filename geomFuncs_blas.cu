
#include "geomFuncs_blas.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>

void solve_8x8_rm(const double A[64], const double b[8], double x[8]);

inline void daxpy(double Y[8], const double X[8], double a);

int ImproveHomography_Mat(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
    if (data.h_data == NULL)
        return 0;

    SiftPoint *mpts = data.h_data;
    const double limit = thresh * thresh;

    const int numPts = data.numPts;
    double M[64] = {0.0};
    double A[8] = {0.0};
    double X[8] = {0.0};
    double Y[8] = {0.0};
    const std::size_t M_size = sizeof(double) * 64;
    const std::size_t X_size = sizeof(double) * 8;
    int numfit = 0;

    // for (int i = 0; i < 8; i++)
    //     A[i] = homography[i] / homography[8];
    A[0] = homography[0] / homography[8];
    A[1] = homography[1] / homography[8];
    A[2] = homography[2] / homography[8];
    A[3] = homography[3] / homography[8];
    A[4] = homography[4] / homography[8];
    A[5] = homography[5] / homography[8];
    A[6] = homography[6] / homography[8];
    A[7] = homography[7] / homography[8];

    for (int loop = 0; loop < numLoops; loop++)
    {
        // for (int i = 0; i < 64; i++)
        //     M[i] = 0.0;
        // for (int i = 0; i < 8; i++)
        //     X[i] = 0;
        std::memset((void *)M, 0, M_size);
        std::memset((void *)X, 0, X_size);

        for (int i = 0; i < numPts; i++)
        {
            SiftPoint &pt = mpts[i];
            if (pt.score < minScore || pt.ambiguity > maxAmbiguity)
                continue;
            double den = A[6] * pt.xpos + A[7] * pt.ypos + 1.0f;
            double dx = (A[0] * pt.xpos + A[1] * pt.ypos + A[2]) / den - pt.match_xpos;
            double dy = (A[3] * pt.xpos + A[4] * pt.ypos + A[5]) / den - pt.match_ypos;
            double err = dx * dx + dy * dy;
            double wei = (err < limit ? 1.0f : 0.0f); // limit / (err + limit);
            Y[0] = pt.xpos;
            Y[1] = pt.ypos;
            Y[2] = 1.0;
            Y[3] = Y[4] = Y[5] = 0.0;
            Y[6] = -pt.xpos * pt.match_xpos;
            Y[7] = -pt.ypos * pt.match_xpos;

            for (int j = 0; j < 64; j++)
                M[j] += (Y[j % 8] * Y[j / 8] * wei);

            daxpy(X, Y, pt.match_xpos * wei);

            Y[0] = Y[1] = Y[2] = 0.0;
            Y[3] = pt.xpos;
            Y[4] = pt.ypos;
            Y[5] = 1.0;
            Y[6] = -pt.xpos * pt.match_ypos;
            Y[7] = -pt.ypos * pt.match_ypos;
            for (int j = 0; j < 64; j++)
                M[j] += (Y[j % 8] * Y[j / 8] * wei);
            daxpy(X, Y, pt.match_ypos * wei);
        }

        solve_8x8_rm(M, X, A);
    }

#pragma omp parallel for reduction(+ : numfit)
    for (int i = 0; i < numPts; i++)
    { // This should probably be moved to cuda kernel
        SiftPoint &pt = mpts[i];
        double den = A[6] * pt.xpos + A[7] * pt.ypos + 1.0;
        double dx = (A[0] * pt.xpos + A[1] * pt.ypos + A[2]) / den - pt.match_xpos;
        double dy = (A[3] * pt.xpos + A[4] * pt.ypos + A[5]) / den - pt.match_ypos;
        double err = dx * dx + dy * dy;
        if (err < limit)
            numfit++;
        pt.match_error = sqrt(err);
    }

    // for (int i = 0; i < 8; i++)
    //     homography[i] = A[i];
    homography[0] = A[0];
    homography[1] = A[1];
    homography[2] = A[2];
    homography[3] = A[3];
    homography[4] = A[4];
    homography[5] = A[5];
    homography[6] = A[6];
    homography[7] = A[7];
    homography[8] = 1.0f;
    return numfit;
}

inline void daxpy(double Y[8], const double X[8], double a)
{
    Y[0] = X[0] * a + Y[0];
    Y[1] = X[1] * a + Y[1];
    Y[2] = X[2] * a + Y[2];
    Y[3] = X[3] * a + Y[3];
    Y[4] = X[4] * a + Y[4];
    Y[5] = X[5] * a + Y[5];
    Y[6] = X[6] * a + Y[6];
    Y[7] = X[7] * a + Y[7];
}

/* Function Definitions */
void solve_8x8_rm(const double A[64], const double b[8], double x[8])
{
    double b_A[64];
    double smax;
    int b_i;
    int i;
    int j;
    int jA;
    int k;
    signed char ipiv[8];
    /*  Solve the following matrix equation for Ax=b */
    for (i = 0; i < 8; i++)
    {
        jA = i << 3;
        b_A[jA] = A[i];
        b_A[jA + 1] = A[i + 8];
        b_A[jA + 2] = A[i + 16];
        b_A[jA + 3] = A[i + 24];
        b_A[jA + 4] = A[i + 32];
        b_A[jA + 5] = A[i + 40];
        b_A[jA + 6] = A[i + 48];
        b_A[jA + 7] = A[i + 56];
        x[i] = b[i];
        ipiv[i] = (signed char)(i + 1);
    }
    for (j = 0; j < 7; j++)
    {
        int b_tmp;
        int jp1j;
        int mmj_tmp;
        signed char i1;
        mmj_tmp = 6 - j;
        b_tmp = j * 9;
        jp1j = b_tmp + 2;
        jA = 8 - j;
        i = 0;
        smax = fabs(b_A[b_tmp]);
        for (k = 2; k <= jA; k++)
        {
            double s;
            s = fabs(b_A[(b_tmp + k) - 1]);
            if (s > smax)
            {
                i = k - 1;
                smax = s;
            }
        }
        if (b_A[b_tmp + i] != 0.0)
        {
            if (i != 0)
            {
                jA = j + i;
                ipiv[j] = (signed char)(jA + 1);
                smax = b_A[j];
                b_A[j] = b_A[jA];
                b_A[jA] = smax;
                smax = b_A[j + 8];
                b_A[j + 8] = b_A[jA + 8];
                b_A[jA + 8] = smax;
                smax = b_A[j + 16];
                b_A[j + 16] = b_A[jA + 16];
                b_A[jA + 16] = smax;
                smax = b_A[j + 24];
                b_A[j + 24] = b_A[jA + 24];
                b_A[jA + 24] = smax;
                smax = b_A[j + 32];
                b_A[j + 32] = b_A[jA + 32];
                b_A[jA + 32] = smax;
                smax = b_A[j + 40];
                b_A[j + 40] = b_A[jA + 40];
                b_A[jA + 40] = smax;
                smax = b_A[j + 48];
                b_A[j + 48] = b_A[jA + 48];
                b_A[jA + 48] = smax;
                smax = b_A[j + 56];
                b_A[j + 56] = b_A[jA + 56];
                b_A[jA + 56] = smax;
            }
            b_i = (b_tmp - j) + 8;
            for (i = jp1j; i <= b_i; i++)
            {
                b_A[i - 1] /= b_A[b_tmp];
            }
        }
        jA = b_tmp;
        for (i = 0; i <= mmj_tmp; i++)
        {
            smax = b_A[(b_tmp + (i << 3)) + 8];
            if (smax != 0.0)
            {
                b_i = jA + 10;
                jp1j = (jA - j) + 16;
                for (k = b_i; k <= jp1j; k++)
                {
                    b_A[k - 1] += b_A[((b_tmp + k) - jA) - 9] * -smax;
                }
            }
            jA += 8;
        }
        i1 = ipiv[j];
        if (i1 != j + 1)
        {
            smax = x[j];
            x[j] = x[i1 - 1];
            x[i1 - 1] = smax;
        }
    }
    for (k = 0; k < 8; k++)
    {
        jA = k << 3;
        if (x[k] != 0.0)
        {
            b_i = k + 2;
            for (i = b_i; i < 9; i++)
            {
                x[i - 1] -= x[k] * b_A[(i + jA) - 1];
            }
        }
    }
    for (k = 7; k >= 0; k--)
    {
        jA = k << 3;
        smax = x[k];
        if (smax != 0.0)
        {
            smax /= b_A[k + jA];
            x[k] = smax;
            for (i = 0; i < k; i++)
            {
                x[i] -= x[k] * b_A[i + jA];
            }
        }
    }
}

/* End of code generation (solve_8x8_rm.c) */
