
#include <complex.h>
#include <math.h>

#include "nfft3.h"
#include <fftw3.h>

#include <stdio.h>


typedef double R;
typedef double _Complex C;
// typedef fftw_complex C;
typedef NFFT_INT INT;

#define K(x) x
#define EXP exp
#define POW pow
#define FABS fabs
#define SQRT sqrt
#define CREAL creal

typedef struct fastsum_plan_
{
    /** api */
    
    int d;          /**< number of dimensions            */
    R sigma;        /**< gaussian kernel parameter       */
    
    int N_total;    /**< number of source & target knots          */
    R *x;           /**< source & target knots in d-ball with radius 1/4-eps/2 */
    C *alpha;       /**< source coefficients             */
    C *f;           /**< target evaluations              */
    
    /** internal */
    int n;          /**< expansion degree                */
    C *b;           /**< expansion coefficients          */
    C *f_hat;       /**< Fourier coefficients of nfft plans */
  
    int p;          /**< degree of smoothness of regularization */
    R eps;          /**< outer boundary                  */
  
    nfft_plan mv1;  /**< source nfft plan */
    nfft_plan mv2;  /**< target nfft plan */
    
} fastsum_plan;


C gaussian(R x, int der, const R c)    /* K(x)=EXP(-x^2/c^2) */
{
    R value = 0.0;
      
    switch (der)
    {
        case  0 : value = EXP(-x*x/(c*c)); break;
        case  1 : value = -K(2.0)*x/(c*c)*EXP(-x*x/(c*c)); break;
        case  2 : value = K(2.0)*EXP(-x*x/(c*c))*(-c*c+K(2.0)*x*x)/(c*c*c*c); break;
        case  3 : value = -K(4.0)*x*EXP(-x*x/(c*c))*(-K(3.0)*c*c+K(2.0)*x*x)/(c*c*c*c*c*c); break;
        case  4 : value = K(4.0)*EXP(-x*x/(c*c))*(K(3.0)*c*c*c*c-K(12.0)*c*c*x*x+K(4.0)*x*x*x*x)/(c*c*c*c*c*c*c*c); break;
        case  5 : value = -K(8.0)*x*EXP(-x*x/(c*c))*(K(15.0)*c*c*c*c-K(20.0)*c*c*x*x+K(4.0)*x*x*x*x)/POW(c,K(10.0)); break;
        case  6 : value = K(8.0)*EXP(-x*x/(c*c))*(-K(15.0)*c*c*c*c*c*c+K(90.0)*x*x*c*c*c*c-K(60.0)*x*x*x*x*c*c+K(8.0)*x*x*x*x*x*x)/POW(c,K(12.0)); break;
        case  7 : value = -K(16.0)*x*EXP(-x*x/(c*c))*(-K(105.0)*c*c*c*c*c*c+K(210.0)*x*x*c*c*c*c-K(84.0)*x*x*x*x*c*c+K(8.0)*x*x*x*x*x*x)/POW(c,K(14.0)); break;
        case  8 : value = K(16.0)*EXP(-x*x/(c*c))*(K(105.0)*c*c*c*c*c*c*c*c-K(840.0)*x*x*c*c*c*c*c*c+K(840.0)*x*x*x*x*c*c*c*c-K(224.0)*x*x*x*x*x*x*c*c+K(16.0)*x*x*x*x*x*x*x*x)/POW(c,K(16.0)); break;
        case  9 : value = -K(32.0)*x*EXP(-x*x/(c*c))*(K(945.0)*c*c*c*c*c*c*c*c-K(2520.0)*x*x*c*c*c*c*c*c+K(1512.0)*x*x*x*x*c*c*c*c-K(288.0)*x*x*x*x*x*x*c*c+K(16.0)*x*x*x*x*x*x*x*x)/POW(c,K(18.0)); break;
        case 10 : value = K(32.0)*EXP(-x*x/(c*c))*(-K(945.0)*POW(c,K(10.0))+K(9450.0)*x*x*c*c*c*c*c*c*c*c-K(12600.0)*x*x*x*x*c*c*c*c*c*c+K(5040.0)*x*x*x*x*x*x*c*c*c*c-K(720.0)*x*x*x*x*x*x*x*x*c*c+K(32.0)*POW(x,K(10.0)))/POW(c,K(20.0)); break;
        case 11 : value = -K(64.0)*x*EXP(-x*x/(c*c))*(-K(10395.0)*POW(c,K(10.0))+K(34650.0)*x*x*c*c*c*c*c*c*c*c-K(27720.0)*x*x*x*x*c*c*c*c*c*c+K(7920.0)*x*x*x*x*x*x*c*c*c*c-K(880.0)*x*x*x*x*x*x*x*x*c*c+K(32.0)*POW(x,K(10.0)))/POW(c,K(22.0)); break;
        case 12 : value = K(64.0)*EXP(-x*x/(c*c))*(K(10395.0)*POW(c,K(12.0))-K(124740.0)*x*x*POW(c,K(10.0))+K(207900.0)*x*x*x*x*c*c*c*c*c*c*c*c-K(110880.0)*x*x*x*x*x*x*c*c*c*c*c*c+K(23760.0)*x*x*x*x*x*x*x*x*c*c*c*c-K(2112.0)*POW(x,K(10.0))*c*c+K(64.0)*POW(x,K(12.0)))/POW(c,K(24.0)); break;
        default : value = K(0.0);
    }
      
    return value;
}


/** factorial */
static double fak(int n) {
    if (n <= 1)
        return 1.0;
    else
        return n * fak(n - 1);
}

/** binomial coefficient */
static double binom(int n, int m)
{
    return fak(n) / fak(m) / fak(n - m);
}

/** basis polynomial for regularized kernel */
static double BasisPoly(int m, int r, double xx)
{
    int k;
    double sum = 0.0;

    for (k = 0; k <= m - r; k++) {
        sum += binom(m + k, k) * POW((xx + K(1.0)) / K(2.0), (R) k);
    }
    return sum * POW((xx + 1.0), (double) r) * POW(1.0 - xx, (double) (m + 1)) / (double)(1 << (m + 1)) / fak(r);
}


static C regkern3(R xx, int p, R sigma, R eps)
{
    int r;
    C sum = 0.0;
    
    xx = FABS(xx);
    
    if (xx >= 0.5) {
        /*return kern(typ,c,0,K(0.5));*/
        xx = 0.5;
    }
    /* else */
    if (xx <= 0.5 - eps)
    {
        return gaussian(xx, 0, sigma);
    }
    else
    {
      sum = gaussian(0.5, 0, sigma) * BasisPoly(p-1, 0, (-2.0 * xx + 1.0 - eps) / eps);
      for (r = 0; r < p; r++)
      {
        sum += POW(0.5*eps, (double) r) * gaussian(0.5-eps, r, sigma) * BasisPoly(p-1, r, (2.0 * xx - 1.0 + eps) / eps);
      }
      return sum;
    }
    return 0.0;
}



void fastsum_init_guru_kernel(fastsum_plan *ths, R sigma, int d, int nn, int p, R eps) {

    // fastsum_init_guru_kernel

    int t,j,k;
    int N[d];
    int n_total;
    fftw_plan fft_plan;
    
    ths->d = d;
    ths->p = p;
    ths->eps = eps;
    
    ths->sigma = sigma;
    
    n_total = 1;
    for (t=0; t<d; ++t) {
        N[t] = nn;
        n_total *= nn;
    }
    
    ths->n = nn;
    ths->b = (C*) nfft_malloc((size_t)(n_total) * sizeof(C));
    ths->f_hat = (C*) nfft_malloc((size_t)(n_total) * sizeof(C));

    fft_plan = fftw_plan_dft(d, N, ths->b, ths->b, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // fastsum_precompute_kernel
    
    for (j = 0; j < n_total; j++)
    {
        k = j;
        ths->b[j] = 0.0;
        for (t = 0; t < d; t++) {
            ths->b[j] += ((double) (k % nn) / (double)nn - 0.5) * ((double) (k % nn) / (double)nn - 0.5);
            k = k / nn;
        }
        ths->b[j] = regkern3(SQRT(CREAL(ths->b[j])), p, ths->sigma, ths->eps) / (R)(n_total);
    }

    nfft_fftshift_complex_int(ths->b, d, N);
    fftw_execute(fft_plan);
    nfft_fftshift_complex_int(ths->b, d, N);
    
    fftw_destroy_plan(fft_plan);
    
    ths->x = NULL;
    ths->alpha = NULL;
    ths->f = NULL;
}



void fastsum_init_guru_nodes(fastsum_plan *ths, int N_total, int nn_oversampled, int m)
{
    int t;
    int N[ths->d], n[ths->d];
    unsigned sort_flags = 0U;
    
    if (ths->d > 1)
    {
        sort_flags = NFFT_SORT_NODES;
    }
    
    ths->N_total = N_total;
    
    ths->x = (R *) nfft_malloc((size_t)(ths->d * N_total) * (sizeof(R)));
    ths->alpha = (C *) nfft_malloc((size_t)(N_total) * (sizeof(C)));
    ths->f = (C *) nfft_malloc((size_t)(N_total) * (sizeof(C)));
    
    /** init d-dimensional NFFT plan */
    for (t = 0; t < ths->d; t++)
    {
        N[t] = ths->n;
        n[t] = nn_oversampled;
    }
    
    nfft_init_guru(&(ths->mv1), ths->d, N, N_total, n, m,
        sort_flags | PRE_PHI_HUT | PRE_PSI | FFTW_INIT | ((ths->d == 1) ? FFT_OUT_OF_PLACE : 0U),
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    ths->mv1.x = ths->x;
    ths->mv1.f = ths->alpha;
    ths->mv1.f_hat = ths->f_hat;
    
    nfft_init_guru(&(ths->mv2), ths->d, N, N_total, n, m,
        sort_flags | PRE_PHI_HUT | PRE_PSI | FFTW_INIT | ((ths->d == 1) ? FFT_OUT_OF_PLACE : 0U),
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    ths->mv2.x = ths->x;
    ths->mv2.f = ths->f;
    ths->mv2.f_hat = ths->f_hat;
}

void fastsum_finalize_kernel(fastsum_plan *ths)
{
    nfft_free(ths->b);
    nfft_free(ths->f_hat);
}

void fastsum_finalize_nodes(fastsum_plan *ths)
{
    nfft_free(ths->x);
    nfft_free(ths->alpha);
    nfft_free(ths->f);
    
    nfft_finalize(&(ths->mv1));
    nfft_finalize(&(ths->mv2));
}

void fastsum_finalize(fastsum_plan *ths) {
    fastsum_finalize_kernel(ths);
    fastsum_finalize_nodes(ths);
}


/** precomputation for fastsum */
void fastsum_precompute(fastsum_plan *ths)
{
    if (ths->mv1.flags & PRE_LIN_PSI)
        nfft_precompute_lin_psi(&(ths->mv1));
    
    if (ths->mv1.flags & PRE_PSI)
        nfft_precompute_psi(&(ths->mv1));
    
    if (ths->mv1.flags & PRE_FULL_PSI)
        nfft_precompute_full_psi(&(ths->mv1));
    
    if (ths->mv2.flags & PRE_LIN_PSI)
        nfft_precompute_lin_psi(&(ths->mv2));

    if (ths->mv2.flags & PRE_PSI)
        nfft_precompute_psi(&(ths->mv2));
    
    if (ths->mv2.flags & PRE_FULL_PSI)
        nfft_precompute_full_psi(&(ths->mv2));
}

void fastsum_trafo(fastsum_plan *ths)
{
    int j, k;
    
    /** first step of algorithm */
    nfft_adjoint(&(ths->mv1));
    
    /** second step of algorithm */
    for (k = 0; k < ths->mv2.N_total; k++)
        ths->mv2.f_hat[k] = ths->b[k] * ths->mv1.f_hat[k];
      
    /** third step of algorithm */
    nfft_trafo(&(ths->mv2));
    
    /** write far field to output */
    for (j = 0; j < ths->N_total; j++)
        ths->f[j] = ths->mv2.f[j];

}

/** direct computation of sums */
void fastsum_exact(fastsum_plan *ths)
{
    int j, k;
    int t;
    R r;
    
    for (j = 0; j < ths->N_total; j++)
    {
        ths->f[j] = K(0.0);
        for (k = 0; k < ths->N_total; k++)
        {
            if (ths->d == 1)
                r = ths->x[j] - ths->x[k];
            else
            {
                r = K(0.0);
                for (t = 0; t < ths->d; t++)
                    r += (ths->x[j * ths->d + t] - ths->x[k * ths->d + t])
                        * (ths->x[j * ths->d + t] - ths->x[k * ths->d + t]);
                r = SQRT(r);
            }
            ths->f[j] += ths->alpha[k] * gaussian(r, 0, ths->sigma);
        }
    }
}