#include <complex>
// started as C code
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define AVX2 1
#if AVX2
  #include <immintrin.h>
  #define malloc(x) aligned_alloc(32, x)

  #define block_ft(B_re, B_im) {\
    __m256d prop_prod_re_vec = _mm256_set1_pd(prop_prod_re[jCprime * Ns + jSprime]); \
    __m256d prop_prod_im_vec = _mm256_set1_pd(prop_prod_im[jCprime * Ns + jSprime]); \
    __m256d * B_re_offset = (__m256d *) (B_re + Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,0,Nc,Ns,Nsrc_f)); \
    __m256d * B_im_offset = (__m256d *) (B_im + Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,0,Nc,Ns,Nsrc_f)); \
    __m256d * psi_re_offset = (__m256d *) (psi_re + index_2d(y, 0, Nsrc_f)); \
    __m256d * psi_im_offset = (__m256d *) (psi_im + index_2d(y, 0, Nsrc_f)); \
    for (m = 0; m < Nsrc / 4; m ++) { \
      B_re_offset[m] += psi_re_offset[m] * prop_prod_re_vec; \
      B_re_offset[m] -= psi_im_offset[m] * prop_prod_im_vec; \
      B_im_offset[m] += psi_re_offset[m] * prop_prod_im_vec; \
      B_im_offset[m] += psi_im_offset[m] * prop_prod_re_vec; \
    } \
  }
#else
  #define block_ft(B_re, B_im) { \
    double prop_prod_re_val = prop_prod_re[jCprime * Ns + jSprime]; \
    double prop_prod_im_val = prop_prod_im[jCprime * Ns + jSprime]; \
    for (m=0; m<Nsrc; m++) { \
       B_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc,Ns,Nsrc_f)] \
         += psi_re[index_2d(y,m ,Nsrc_f)] * prop_prod_re_val - psi_im[index_2d(y,m ,Nsrc_f)] * prop_prod_im_val; \
       B_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc,Ns,Nsrc_f)] \
         += psi_im[index_2d(y,m ,Nsrc_f)] * prop_prod_re_val + psi_re[index_2d(y,m ,Nsrc_f)] * prop_prod_im_val; \
    } \
  }
#endif

int index_2d(int a, int b, int length2) {
   return b +length2*( a );
}
int index_3d(int a, int b, int c, int length2, int length3) {
   return c +length3*( b +length2*( a ));
}
int index_4d(int a, int b, int c, int d, int length2, int length3, int length4) {
   return d +length4*( c +length3*( b +length2*( a )));
}
int index_5d(int a, int b, int c, int d, int e, int length2, int length3, int length4, int length5) {
   return e + length5*( d +length4*( c +length3*( b +length2*( a ))));
}
int prop_index(int q, int t, int c1, int s1, int c2, int s2, int y, int x, int Nc_f, int Ns_f, int Vsrc_f, int Vsnk_f, int Nt_f) {
   return y +Vsrc_f*( x +Vsnk_f*( s1 +Ns_f*( c1 +Nc_f*( s2 +Ns_f*( c2 +Nc_f*( t +Nt_f* q ))))));
}
int id_prop_index(int t, int c1, int s1, int c2, int s2, int y, int x, int Nc_f, int Ns_f, int Vsrc_f, int Vsnk_f, int Nt_f) {
   return y +Vsrc_f*( x +Vsnk_f*( s1 +Ns_f*( c1 +Nc_f*( s2 +Ns_f*( c2 +Nc_f*( t ))))));
}
int Blocal_index(int c1, int s1, int c2, int s2, int c3, int s3, int m, int Nc_f, int Ns_f, int Nsrc_f) {
   return m +Nsrc_f*( s3 +Ns_f*( c3 +Nc_f*( s2 +Ns_f*( c2 +Nc_f*( s1 +Ns_f*( c1 ))))));
}

#define zero_block(B) { \
   for (iCprime=0; iCprime<Nc_f; iCprime++) \
      for (iSprime=0; iSprime<Ns_f; iSprime++) \
         for (kCprime=0; kCprime<Nc_f; kCprime++) \
            for (kSprime=0; kSprime<Ns_f; kSprime++) \
               for (jCprime=0; jCprime<Nc_f; jCprime++) \
                  for (jSprime=0; jSprime<Ns_f; jSprime++) \
                     for (m=0; m<Nsrc_f; m++) \
                        B[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] = 0.0; \
}

// timers
clock_t local_block_time;
clock_t first_block_time;
clock_t second_block_time;
clock_t third_block_time;
clock_t correlator_time;
clock_t total_time;

void make_local_block(double* Blocal_re, 
    double* Blocal_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int x,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f) {
   local_block_time -= clock();
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, y, wnum, m;
   /* subexpressions */
   std::complex<double> prop_prod_02;
   std::complex<double> prop_prod;
   /* initialize */
   zero_block(Blocal_re);
   zero_block(Blocal_im);
   /* build local (no quark exchange) block */
   for (wnum=0; wnum<Nw_f; wnum++) {
      iC = color_weights[index_2d(wnum,0, Nq_f)];
      iS = spin_weights[index_2d(wnum,0, Nq_f)];
      jC = color_weights[index_2d(wnum,1, Nq_f)];
      jS = spin_weights[index_2d(wnum,1, Nq_f)];
      kC = color_weights[index_2d(wnum,2, Nq_f)];
      kS = spin_weights[index_2d(wnum,2, Nq_f)];
      for (iCprime=0; iCprime<Nc_f; iCprime++) {
         for (iSprime=0; iSprime<Ns_f; iSprime++) {
            for (kCprime=0; kCprime<Nc_f; kCprime++) {
               for (kSprime=0; kSprime<Ns_f; kSprime++) {
                  for (y=0; y<Vsrc_f; y++) {
                     std::complex<double> prop_0(prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2(prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     prop_prod_02 = weights[wnum] * ( prop_0 * prop_2 );
                     for (jCprime=0; jCprime<Nc_f; jCprime++) {
                        for (jSprime=0; jSprime<Ns_f; jSprime++) {
                           std::complex<double> prop_1(prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                           prop_prod = prop_prod_02 * prop_1;
                           for (m=0; m<Nsrc_f; m++) {
                              std::complex<double> psi(psi_re[index_2d(y,m ,Nsrc_f)], psi_im[index_2d(y,m ,Nsrc_f)]);
                              std::complex<double> block = psi * prop_prod;
                              Blocal_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += real(block);
                              Blocal_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += imag(block);
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   local_block_time += clock();
}

void make_local_snk_block(double* Blocal_re, 
    double* Blocal_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int y,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsnk_f) {
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, wnum, n;
   /* subexpressions */
   std::complex<double> prop_prod_02;
   std::complex<double> prop_prod;
   /* initialize */
   int Nsrc_f = Nsnk_f;
   int m;
   zero_block(Blocal_re);
   zero_block(Blocal_im);
   /* build local (no quark exchange) block */
   for (wnum=0; wnum<Nw_f; wnum++) {
      iC = color_weights[index_2d(wnum,0, Nq_f)];
      iS = spin_weights[index_2d(wnum,0, Nq_f)];
      jC = color_weights[index_2d(wnum,1, Nq_f)];
      jS = spin_weights[index_2d(wnum,1, Nq_f)];
      kC = color_weights[index_2d(wnum,2, Nq_f)];
      kS = spin_weights[index_2d(wnum,2, Nq_f)];
      for (iCprime=0; iCprime<Nc_f; iCprime++) {
         for (iSprime=0; iSprime<Ns_f; iSprime++) {
            for (kCprime=0; kCprime<Nc_f; kCprime++) {
               for (kSprime=0; kSprime<Ns_f; kSprime++) {
                  for (x=0; x<Vsnk_f; x++) {
                     std::complex<double> prop_0(prop_re[prop_index(0,t,iCprime,iSprime,iC,iS,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iCprime,iSprime,iC,iS,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2(prop_re[prop_index(2,t,kCprime,kSprime,kC,kS,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kCprime,kSprime,kC,kS,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     prop_prod_02 = weights[wnum] * ( prop_0 * prop_2 );
                     for (jCprime=0; jCprime<Nc_f; jCprime++) {
                        for (jSprime=0; jSprime<Ns_f; jSprime++) {
                           std::complex<double> prop_1(prop_re[prop_index(1,t,jCprime,jSprime,jC,jS,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(1,t,jCprime,jSprime,jC,jS,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                           prop_prod = prop_prod_02 * prop_1;
                           for (n=0; n<Nsnk_f; n++) {
                              std::complex<double> psi(psi_re[index_2d(x,n ,Nsnk_f)], psi_im[index_2d(x,n ,Nsnk_f)]);
                              std::complex<double> block = psi * prop_prod;
                              Blocal_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,n ,Nc_f,Ns_f,Nsnk_f)] += real(block);
                              Blocal_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,n ,Nc_f,Ns_f,Nsnk_f)] += imag(block);
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void make_second_block(double* Bsecond_re, 
    double* Bsecond_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int x1,
    const int x2,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f) {
   second_block_time -= clock();
   assert(Nc == Nc_f);
   assert(Ns == Ns_f);
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, y, wnum, m;
   /* subexpressions */
   std::complex <double> prop_prod_02;
   double prop_prod_1_re [Ns * Nc], prop_prod_1_im [Ns * Nc], 
          prop_prod_2_re [Ns * Nc], prop_prod_2_im [Ns * Nc];
   /* initialize */
   zero_block(Bsecond_re);
   zero_block(Bsecond_im);
   double packed_prop_1_re [Vsrc_f * Nc * Ns * Nc * Ns], packed_prop_1_im [Vsrc_f * Nc * Ns * Nc * Ns];
   for (y = 0; y < Vsrc_f; y ++)
      for (jC = 0; jC < Nc; jC ++)
         for (jS = 0; jS < Ns; jS ++)
            for (jCprime = 0; jCprime < Nc; jCprime ++)
               for (jSprime = 0; jSprime < Ns; jSprime ++) {
                  int new_index = (((y * Nc + jC) * Ns + jS) * Nc + jCprime) * Ns + jSprime;
                  int old_index = prop_index(1,t,jC,jS,jCprime,jSprime,y,x2 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f);
                  packed_prop_1_re[new_index] = prop_re[old_index];
                  packed_prop_1_im[new_index] = prop_im[old_index];
               }
   /* build local (no quark exchange) block */
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (kCprime=0; kCprime<Nc; kCprime++) {
            for (kSprime=0; kSprime<Ns; kSprime++) {
               for (y=0; y<Vsrc_f; y++) {
                  for (int index = 0; index < Nc * Ns; index ++) {
                     prop_prod_1_re[index] = prop_prod_1_im[index] = 0;
                     prop_prod_2_re[index] = prop_prod_2_im[index] = 0;
                  }
                  for (wnum=0; wnum<Nw_f; wnum++) {
                     iC = color_weights[index_2d(wnum,0, Nq_f)];
                     iS = spin_weights[index_2d(wnum,0, Nq_f)];
                     jC = color_weights[index_2d(wnum,1, Nq_f)];
                     jS = spin_weights[index_2d(wnum,1, Nq_f)];
                     kC = color_weights[index_2d(wnum,2, Nq_f)];
                     kS = spin_weights[index_2d(wnum,2, Nq_f)];
                     std::complex<double> prop_0(prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2(prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f)]);
                     prop_prod_02 = weights[wnum] * ( prop_0 * prop_2 );
                     double prop_prod_02_re = real(prop_prod_02);
                     double prop_prod_02_im = imag(prop_prod_02);
                     for (jCprime=0; jCprime<Nc; jCprime++) {
                        for (jSprime=0; jSprime<Ns; jSprime++) {
                           int packed_index = (((y * Nc + jC) * Ns + jS) * Nc + jCprime) * Ns + jSprime;
                           double prop_1_re = packed_prop_1_re[packed_index];
                           double prop_1_im = packed_prop_1_im[packed_index];
                           prop_prod_1_re[jCprime * Ns + jSprime] += prop_1_re * prop_prod_02_re - prop_1_im * prop_prod_02_im;
                           prop_prod_1_im[jCprime * Ns + jSprime] += prop_1_im * prop_prod_02_re + prop_1_re * prop_prod_02_im;
                        }
                     }
                  }
                  for (jCprime=0; jCprime<Nc; jCprime++) {
                     for (jSprime=0; jSprime<Ns; jSprime++) {
                        double * prop_prod_re, * prop_prod_im;
                        prop_prod_re = prop_prod_1_re; prop_prod_im = prop_prod_1_im;
                        block_ft(Bsecond_re, Bsecond_im);
                     }
                  }
               }
            }
         }
      }
   }
   second_block_time += clock();
}


void make_first_block(double* Bfirst_re, 
    double* Bfirst_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int x1,
    const int x2,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f) {
   first_block_time -= clock();
   assert(Nc == Nc_f);
   assert(Ns == Ns_f);
   assert(Nsrc == Nsrc_f);
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, y, wnum, m;
   /* subexpressions */
   std::complex <double> prop_prod_02;
   double prop_prod_re[Ns * Nc], prop_prod_im[Ns * Nc];
   /* initialize */
   zero_block(Bfirst_re);
   zero_block(Bfirst_im);
   double packed_prop_0_re [Vsrc_f * Nc * Ns * Nc * Ns], packed_prop_0_im [Vsrc_f * Nc * Ns * Nc * Ns];
   double packed_prop_1_re [Vsrc_f * Nc * Ns * Nc * Ns], packed_prop_1_im [Vsrc_f * Nc * Ns * Nc * Ns];
   double packed_prop_2_re [Vsrc_f * Nc * Ns * Nc * Ns], packed_prop_2_im [Vsrc_f * Nc * Ns * Nc * Ns];
   for (y = 0; y < Vsrc_f; y ++)
      for (iC = 0; iC < Nc; iC ++)
         for (iS = 0; iS < Ns; iS ++)
            for (iCprime = 0; iCprime < Nc; iCprime ++)
               for (iSprime = 0; iSprime < Ns; iSprime ++) {
                  int new_index = (((y * Nc + iC) * Ns + iS) * Nc + iCprime) * Ns + iSprime;
                  int old_index = prop_index(0,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f);
                  packed_prop_0_re[new_index] = prop_re[old_index];
                  packed_prop_0_im[new_index] = prop_im[old_index];
                  old_index = prop_index(1,t,iC,iS,iCprime,iSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f);
                  packed_prop_1_re[new_index] = prop_re[old_index];
                  packed_prop_1_im[new_index] = prop_im[old_index];
                  old_index = prop_index(2,t,iC,iS,iCprime,iSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f);
                  packed_prop_2_re[new_index] = prop_re[old_index];
                  packed_prop_2_im[new_index] = prop_im[old_index];
               }
   /* build local (no quark exchange) block */
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (kCprime=0; kCprime<Nc; kCprime++) {
            for (kSprime=0; kSprime<Ns; kSprime++) {
               for (y=0; y<Vsrc_f; y++) {
                  for (int index = 0; index < Nc * Ns; index ++)
                     prop_prod_re[index] = prop_prod_im[index] = 0;
                  for (wnum=0; wnum<Nw_f; wnum++) {
                     iC = color_weights[index_2d(wnum,0, Nq_f)];
                     iS = spin_weights[index_2d(wnum,0, Nq_f)];
                     jC = color_weights[index_2d(wnum,1, Nq_f)];
                     jS = spin_weights[index_2d(wnum,1, Nq_f)];
                     kC = color_weights[index_2d(wnum,2, Nq_f)];
                     kS = spin_weights[index_2d(wnum,2, Nq_f)];
                     int prop_0_index = (((y * Nc + iC) * Ns + iS) * Nc + iCprime) * Ns + iSprime;
                     int prop_2_index = (((y * Nc + kC) * Ns + kS) * Nc + kCprime) * Ns + kSprime;
                     double prop_0_re = packed_prop_0_re[prop_0_index];
                     double prop_0_im = packed_prop_0_im[prop_0_index];
                     double prop_2_re = packed_prop_2_re[prop_2_index];
                     double prop_2_im = packed_prop_2_im[prop_2_index];
                     double prop_prod_02_re = weights[wnum] * (prop_0_re * prop_2_re - prop_0_im * prop_2_im);
                     double prop_prod_02_im = weights[wnum] * (prop_0_im * prop_2_re + prop_0_re * prop_2_im);
                     for (jCprime=0; jCprime<Nc; jCprime++) {
                        for (jSprime=0; jSprime<Ns; jSprime++) {
                           int packed_index = (((y * Nc + jC) * Ns + jS) * Nc + jCprime) * Ns + jSprime;
                           double prop_1_re = packed_prop_1_re[packed_index];
                           double prop_1_im = packed_prop_1_im[packed_index];
                           prop_prod_re[jCprime * Ns + jSprime] += prop_1_re * prop_prod_02_re - prop_1_im * prop_prod_02_im;
                           prop_prod_im[jCprime * Ns + jSprime] += prop_1_im * prop_prod_02_re + prop_1_re * prop_prod_02_im;
                        }
                     }
                  }
                  for (jCprime=0; jCprime<Nc; jCprime++) {
                     for (jSprime=0; jSprime<Ns; jSprime++) {
                        block_ft(Bfirst_re, Bfirst_im);
                     }
                  }
               }
            }
         }
      }
   }
   first_block_time += clock();
}

void make_third_block(double* Bthird_re, 
    double* Bthird_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int x1,
    const int x2,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f) {
   third_block_time -= clock();
   assert(Nc == Nc_f);
   assert(Ns == Ns_f);
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, y, wnum, m;
   /* subexpressions */
   std::complex <double> prop_prod_02;
   double prop_prod_re[Ns * Nc], prop_prod_im[Ns * Nc];
   /* initialize */
   zero_block(Bthird_re);
   zero_block(Bthird_im);
   double packed_prop_1_re [Vsrc_f * Nc * Ns * Nc * Ns], packed_prop_1_im [Vsrc_f * Nc * Ns * Nc * Ns];
   for (y = 0; y < Vsrc_f; y ++)
      for (jC = 0; jC < Nc; jC ++)
         for (jS = 0; jS < Ns; jS ++)
            for (jCprime = 0; jCprime < Nc; jCprime ++)
               for (jSprime = 0; jSprime < Ns; jSprime ++) {
                  int new_index = (((y * Nc + jC) * Ns + jS) * Nc + jCprime) * Ns + jSprime;
                  int old_index = prop_index(1,t,jC,jS,jCprime,jSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f);
                  packed_prop_1_re[new_index] = prop_re[old_index];
                  packed_prop_1_im[new_index] = prop_im[old_index];
               }
   /* build local (no quark exchange) block */
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (kCprime=0; kCprime<Nc; kCprime++) {
            for (kSprime=0; kSprime<Ns; kSprime++) {
               for (y=0; y<Vsrc_f; y++) {
                  for (int index = 0; index < Nc * Ns; index ++)
                     prop_prod_re[index] = prop_prod_im[index] = 0;
                  for (wnum=0; wnum<Nw_f; wnum++) {
                     iC = color_weights[index_2d(wnum,0, Nq_f)];
                     iS = spin_weights[index_2d(wnum,0, Nq_f)];
                     jC = color_weights[index_2d(wnum,1, Nq_f)];
                     jS = spin_weights[index_2d(wnum,1, Nq_f)];
                     kC = color_weights[index_2d(wnum,2, Nq_f)];
                     kS = spin_weights[index_2d(wnum,2, Nq_f)];
                     std::complex<double> prop_0(prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x1 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2(prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x2 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x2 ,Nc,Ns,Vsrc_f,Vsnk_f,Nt_f)]);
                     prop_prod_02 = weights[wnum] * ( prop_0 * prop_2 );
                     double prop_prod_02_re = real(prop_prod_02);
                     double prop_prod_02_im = imag(prop_prod_02);
                     for (jCprime=0; jCprime<Nc; jCprime++) {
                        for (jSprime=0; jSprime<Ns; jSprime++) {
                           int packed_index = (((y * Nc + jC) * Ns + jS) * Nc + jCprime) * Ns + jSprime;
                           double prop_1_re = packed_prop_1_re[packed_index];
                           double prop_1_im = packed_prop_1_im[packed_index];
                           prop_prod_re[jCprime * Ns + jSprime] += prop_1_re * prop_prod_02_re - prop_1_im * prop_prod_02_im;
                           prop_prod_im[jCprime * Ns + jSprime] += prop_1_im * prop_prod_02_re + prop_1_re * prop_prod_02_im;
                        }
                     }
                  }
                  for (jCprime=0; jCprime<Nc; jCprime++) {
                     for (jSprime=0; jSprime<Ns; jSprime++) {
                        double prop_prod_re_val = prop_prod_re[jCprime * Ns + jSprime];
                        double prop_prod_im_val = prop_prod_im[jCprime * Ns + jSprime];
                        block_ft(Bthird_re, Bthird_im);
                     }
                  }
               }
            }
         }
      }
   }
   third_block_time += clock();
}

void make_dibaryon_correlator(double* C_re,
    double* C_im,
    const double* B1_r1_Blocal_re, 
    const double* B1_r1_Blocal_im, 
    const double* B1_r1_Bfirst_re, 
    const double* B1_r1_Bfirst_im, 
    const double* B1_r1_Bsecond_re, 
    const double* B1_r1_Bsecond_im, 
    const double* B1_r1_Bthird_re, 
    const double* B1_r1_Bthird_im, 
    const double* B2_r1_Blocal_re, 
    const double* B2_r1_Blocal_im, 
    const double* B2_r1_Bfirst_re, 
    const double* B2_r1_Bfirst_im, 
    const double* B2_r1_Bsecond_re, 
    const double* B2_r1_Bsecond_im, 
    const double* B2_r1_Bthird_re, 
    const double* B2_r1_Bthird_im, 
    const double* B1_r2_Blocal_re, 
    const double* B1_r2_Blocal_im, 
    const double* B1_r2_Bfirst_re, 
    const double* B1_r2_Bfirst_im, 
    const double* B1_r2_Bsecond_re, 
    const double* B1_r2_Bsecond_im, 
    const double* B1_r2_Bthird_re, 
    const double* B1_r2_Bthird_im, 
    const double* B2_r2_Blocal_re, 
    const double* B2_r2_Blocal_im, 
    const double* B2_r2_Bfirst_re, 
    const double* B2_r2_Bfirst_im, 
    const double* B2_r2_Bsecond_re, 
    const double* B2_r2_Bsecond_im, 
    const double* B2_r2_Bthird_re, 
    const double* B2_r2_Bthird_im, 
    const int* src_spins,
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
    const double* snk_psi_re,
    const double* snk_psi_im,
    const int t,
    const int x1,
    const int x2,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f,
    const int Nsnk_f,
    const int Nperms_f) {
   correlator_time -= clock();
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,wnum,nperm,b,n,m;
   int Nb_f = 2;
   int Nw2_f = Nw_f*Nw_f;
   std::complex<double> term, new_term;
   /* build dibaryon */
   int snk_1_nq[Nb_f];
   int snk_2_nq[Nb_f];
   int snk_3_nq[Nb_f];
   int snk_1_b[Nb_f];
   int snk_2_b[Nb_f];
   int snk_3_b[Nb_f];
   int snk_1[Nb_f];
   int snk_2[Nb_f];
   int snk_3[Nb_f];
   const double * B_re [Nb_f];
   const double * B_im [Nb_f];
   double sum_re[Nsrc_f], sum_im[Nsrc_f];
   for (n = 0; n < Nsrc_f; n ++)
      sum_re[n] = sum_im[n] = 0;
   for (nperm=0; nperm<Nperms_f; nperm++) {
      for (b=0; b<Nb_f; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq_f*b+0 ,2*Nq_f)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq_f*b+1 ,2*Nq_f)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq_f*b+2 ,2*Nq_f)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq_f) / Nq_f;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq_f) / Nq_f;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq_f) / Nq_f;
         snk_1_nq[b] = snk_1[b] % Nq_f;
         snk_2_nq[b] = snk_2[b] % Nq_f;
         snk_3_nq[b] = snk_3[b] % Nq_f;
         if ((src_spins[b] == 1) && (snk_1_b[b] == 0) && (snk_2_b[b] == 0) && (snk_3_b[b] == 0)) {
           B_re[b] = B1_r1_Blocal_re;  B_im[b] = B1_r1_Blocal_im;
         } else if ((src_spins[b] == 1) && (snk_1_b[b] == 1) && (snk_2_b[b] == 1) && (snk_3_b[b] == 1)) {
           B_re[b] = B2_r1_Blocal_re;  B_im[b] = B2_r1_Blocal_im;
         } else if ((src_spins[b] == 1) && (snk_1_b[b] == 0) && (snk_2_b[b] == 1) && (snk_3_b[b] == 0)) {
           B_re[b] = B1_r1_Bsecond_re; B_im[b] = B1_r1_Bsecond_im;
         } else if ((src_spins[b] == 1) && (snk_1_b[b] == 1) && (snk_2_b[b] == 0) && (snk_3_b[b] == 1)) {
           B_re[b] = B2_r1_Bsecond_re; B_im[b] = B2_r1_Bsecond_im;
         } else if ((src_spins[b] == 1) && (snk_1_b[b] == 1) && (snk_2_b[b] == 0) && (snk_3_b[b] == 0)) {
           B_re[b] = B1_r1_Bfirst_re;  B_im[b] = B1_r1_Bfirst_im;
         } else if ((src_spins[b] == 1) && (snk_1_b[b] == 0) && (snk_2_b[b] == 1) && (snk_3_b[b] == 1)) {
           B_re[b] = B2_r1_Bfirst_re;  B_im[b] = B2_r1_Bfirst_im;
         } else if ((src_spins[b] == 1) && (snk_1_b[b] == 0) && (snk_2_b[b] == 0) && (snk_3_b[b] == 1)) {
           B_re[b] = B1_r1_Bthird_re;  B_im[b] = B1_r1_Bthird_im;
         } else if ((src_spins[b] == 1) && (snk_1_b[b] == 1) && (snk_2_b[b] == 1) && (snk_3_b[b] == 0)) {
           B_re[b] = B2_r1_Bthird_re;  B_im[b] = B2_r1_Bthird_im;
         } else if ((src_spins[b] == 2) && (snk_1_b[b] == 0) && (snk_2_b[b] == 0) && (snk_3_b[b] == 0)) {
           B_re[b] = B1_r2_Blocal_re;  B_im[b] = B1_r2_Blocal_im;
         } else if ((src_spins[b] == 2) && (snk_1_b[b] == 1) && (snk_2_b[b] == 1) && (snk_3_b[b] == 1)) {
           B_re[b] = B2_r2_Blocal_re;  B_im[b] = B2_r2_Blocal_im;
         } else if ((src_spins[b] == 2) && (snk_1_b[b] == 0) && (snk_2_b[b] == 1) && (snk_3_b[b] == 0)) {
           B_re[b] = B1_r2_Bsecond_re; B_im[b] = B1_r2_Bsecond_im;
         } else if ((src_spins[b] == 2) && (snk_1_b[b] == 1) && (snk_2_b[b] == 0) && (snk_3_b[b] == 1)) {
           B_re[b] = B2_r2_Bsecond_re; B_im[b] = B2_r2_Bsecond_im;
         } else if ((src_spins[b] == 2) && (snk_1_b[b] == 1) && (snk_2_b[b] == 0) && (snk_3_b[b] == 0)) {
           B_re[b] = B1_r2_Bfirst_re;  B_im[b] = B1_r2_Bfirst_im;
         } else if ((src_spins[b] == 2) && (snk_1_b[b] == 0) && (snk_2_b[b] == 1) && (snk_3_b[b] == 1)) {
           B_re[b] = B2_r2_Bfirst_re;  B_im[b] = B2_r2_Bfirst_im;
         } else if ((src_spins[b] == 2) && (snk_1_b[b] == 0) && (snk_2_b[b] == 0) && (snk_3_b[b] == 1)) {
           B_re[b] = B1_r2_Bthird_re;  B_im[b] = B1_r2_Bthird_im;
         } else if ((src_spins[b] == 2) && (snk_1_b[b] == 1) && (snk_2_b[b] == 1) && (snk_3_b[b] == 0)) {
           B_re[b] = B2_r2_Bthird_re;  B_im[b] = B2_r2_Bthird_im;
         }

      }
      if ((x1 == 0) && (x2 == 0))
         printf("perm %d is %d %d %d %d %d %d, sig %d \n", nperm, perms[index_2d(nperm,0 ,2*Nq_f)] , perms[index_2d(nperm,1 ,2*Nq_f)], perms[index_2d(nperm,2 ,2*Nq_f)], perms[index_2d(nperm,3 ,2*Nq_f)], perms[index_2d(nperm,4 ,2*Nq_f)], perms[index_2d(nperm,5 ,2*Nq_f)], sigs[nperm] );
      for (wnum=0; wnum< Nw2_f; wnum++) {
         iC1 = snk_color_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2_f,Nq_f)];
         iS1 = snk_spin_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2_f,Nq_f)];
         jC1 = snk_color_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2_f,Nq_f)];
         jS1 = snk_spin_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2_f,Nq_f)];
         kC1 = snk_color_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2_f,Nq_f)];
         kS1 = snk_spin_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2_f,Nq_f)];
         iC2 = snk_color_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2_f,Nq_f)];
         iS2 = snk_spin_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2_f,Nq_f)];
         jC2 = snk_color_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2_f,Nq_f)];
         jS2 = snk_spin_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2_f,Nq_f)];
         kC2 = snk_color_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2_f,Nq_f)];
         kS2 = snk_spin_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2_f,Nq_f)]; 
         int index_1 = Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,0,Nc_f,Ns_f,Nsrc_f);
         int index_2 = Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,0,Nc_f,Ns_f,Nsrc_f);
         double prefactor = sigs[nperm] * overall_weight * snk_weights[wnum];
         for (m=0; m < Nsrc_f; m ++) {
            sum_re[m] += prefactor * (B_re[0][index_1 + m] * B_re[1][index_2 + m] - B_im[0][index_1 + m] * B_im[1][index_2 + m]);
            sum_im[m] += prefactor * (B_im[0][index_1 + m] * B_re[1][index_2 + m] + B_re[0][index_1 + m] * B_im[1][index_2 + m]);
         }
      }
   }
   for (int m = 0; m < Nsrc_f; m ++) {
      for (n=0; n<Nsnk_f; n++) {
         double psi_re = snk_psi_re[index_3d(x1,x2,n ,Vsnk_f,Nsnk_f)];
         double psi_im = snk_psi_im[index_3d(x1,x2,n ,Vsnk_f,Nsnk_f)];
         C_re[index_3d(m,n,t,Nsnk_f,Nt_f)] += sum_re[m] * psi_re - sum_im[m] * psi_im;
         C_im[index_3d(m,n,t,Nsnk_f,Nt_f)] += sum_im[m] * psi_re + sum_re[m] * psi_im;
      }
   }
   correlator_time += clock();
}

void make_dibaryon_hex_correlator(double* C_re,
    double* C_im,
    const double* B1_Blocal_re, 
    const double* B1_Blocal_im, 
    const double* B2_Blocal_re, 
    const double* B2_Blocal_im, 
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
    const double* hex_snk_psi_re,
    const double* hex_snk_psi_im,
    const int t,
    const int x,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw2Hex_f,
    const int Nq_f,
    const int Nsrc_f,
    const int Nsnk_fHex,
    const int Nperms_f) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,wnum,nperm,b,n,m;
   int Nb_f = 2;
   double term_re, term_im;
   /* build dibaryon */
   int snk_1_nq[Nb_f];
   int snk_2_nq[Nb_f];
   int snk_3_nq[Nb_f];
   int snk_1_b[Nb_f];
   int snk_2_b[Nb_f];
   int snk_3_b[Nb_f];
   int snk_1[Nb_f];
   int snk_2[Nb_f];
   int snk_3[Nb_f];
   for (nperm=0; nperm<Nperms_f; nperm++) {
      for (b=0; b<Nb_f; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq_f*b+0 ,2*Nq_f)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq_f*b+1 ,2*Nq_f)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq_f*b+2 ,2*Nq_f)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq_f) / Nq_f;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq_f) / Nq_f;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq_f) / Nq_f;
         snk_1_nq[b] = snk_1[b] % Nq_f;
         snk_2_nq[b] = snk_2[b] % Nq_f;
         snk_3_nq[b] = snk_3[b] % Nq_f;
      }
      for (wnum=0; wnum< Nw2Hex_f; wnum++) {
         iC1 = snk_color_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2Hex_f,Nq_f)];
         iS1 = snk_spin_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2Hex_f,Nq_f)];
         jC1 = snk_color_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2Hex_f,Nq_f)];
         jS1 = snk_spin_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2Hex_f,Nq_f)];
         kC1 = snk_color_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2Hex_f,Nq_f)];
         kS1 = snk_spin_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2Hex_f,Nq_f)];
         iC2 = snk_color_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2Hex_f,Nq_f)];
         iS2 = snk_spin_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2Hex_f,Nq_f)];
         jC2 = snk_color_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2Hex_f,Nq_f)];
         jS2 = snk_spin_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2Hex_f,Nq_f)];
         kC2 = snk_color_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2Hex_f,Nq_f)];
         kS2 = snk_spin_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2Hex_f,Nq_f)]; 
         for (m=0; m<Nsrc_f; m++) {
            term_re = sigs[nperm] * overall_weight * snk_weights[wnum] * (B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)] * B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)] - B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)] * B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)]);
            term_im = sigs[nperm] * overall_weight * snk_weights[wnum] * (B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)] * B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)] + B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)] * B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)]);
            for (n=0; n<Nsnk_fHex; n++) {
               C_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] += hex_snk_psi_re[index_2d(x,n ,Nsnk_fHex)] * term_re - hex_snk_psi_im[index_2d(x,n ,Nsnk_fHex)] * term_im;
               C_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] += hex_snk_psi_re[index_2d(x,n ,Nsnk_fHex)] * term_im + hex_snk_psi_im[index_2d(x,n ,Nsnk_fHex)] * term_re;
            }
         }
      }
   } 
}

void make_hex_dibaryon_correlator(double* C_re,
    double* C_im,
    const double* B1_Blocal_re, 
    const double* B1_Blocal_im, 
    const double* B2_Blocal_re, 
    const double* B2_Blocal_im, 
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* src_color_weights, 
    const int* src_spin_weights, 
    const double* src_weights, 
    const double* hex_src_psi_re,
    const double* hex_src_psi_im,
    const int t,
    const int y,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw2Hex_f,
    const int Nq_f,
    const int Nsrc_fHex,
    const int Nsnk_f,
    const int Nperms_f) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,wnum,nperm,b,n,m;
   int Nb_f = 2;
   double term_re, term_im;
   /* build dibaryon */
   int src_1_nq[Nb_f];
   int src_2_nq[Nb_f];
   int src_3_nq[Nb_f];
   int src_1_b[Nb_f];
   int src_2_b[Nb_f];
   int src_3_b[Nb_f];
   int src_1[Nb_f];
   int src_2[Nb_f];
   int src_3[Nb_f];
   for (nperm=0; nperm<Nperms_f; nperm++) {
      for (b=0; b<Nb_f; b++) {
         src_1[b] = perms[index_2d(nperm,Nq_f*b+0 ,2*Nq_f)] - 1;
         src_2[b] = perms[index_2d(nperm,Nq_f*b+1 ,2*Nq_f)] - 1;
         src_3[b] = perms[index_2d(nperm,Nq_f*b+2 ,2*Nq_f)] - 1;
         src_1_b[b] = (src_1[b] - src_1[b] % Nq_f) / Nq_f;
         src_2_b[b] = (src_2[b] - src_2[b] % Nq_f) / Nq_f;
         src_3_b[b] = (src_3[b] - src_3[b] % Nq_f) / Nq_f;
         src_1_nq[b] = src_1[b] % Nq_f;
         src_2_nq[b] = src_2[b] % Nq_f;
         src_3_nq[b] = src_3[b] % Nq_f;
      }
      for (wnum=0; wnum< Nw2Hex_f; wnum++) {
         iC1 = src_color_weights[index_3d(src_1_b[0],wnum,src_1_nq[0] ,Nw2Hex_f,Nq_f)];
         iS1 = src_spin_weights[index_3d(src_1_b[0],wnum,src_1_nq[0] ,Nw2Hex_f,Nq_f)];
         jC1 = src_color_weights[index_3d(src_2_b[0],wnum,src_2_nq[0] ,Nw2Hex_f,Nq_f)];
         jS1 = src_spin_weights[index_3d(src_2_b[0],wnum,src_2_nq[0] ,Nw2Hex_f,Nq_f)];
         kC1 = src_color_weights[index_3d(src_3_b[0],wnum,src_3_nq[0] ,Nw2Hex_f,Nq_f)];
         kS1 = src_spin_weights[index_3d(src_3_b[0],wnum,src_3_nq[0] ,Nw2Hex_f,Nq_f)];
         iC2 = src_color_weights[index_3d(src_1_b[1],wnum,src_1_nq[1] ,Nw2Hex_f,Nq_f)];
         iS2 = src_spin_weights[index_3d(src_1_b[1],wnum,src_1_nq[1] ,Nw2Hex_f,Nq_f)];
         jC2 = src_color_weights[index_3d(src_2_b[1],wnum,src_2_nq[1] ,Nw2Hex_f,Nq_f)];
         jS2 = src_spin_weights[index_3d(src_2_b[1],wnum,src_2_nq[1] ,Nw2Hex_f,Nq_f)];
         kC2 = src_color_weights[index_3d(src_3_b[1],wnum,src_3_nq[1] ,Nw2Hex_f,Nq_f)];
         kS2 = src_spin_weights[index_3d(src_3_b[1],wnum,src_3_nq[1] ,Nw2Hex_f,Nq_f)]; 
         for (n=0; n<Nsnk_f; n++) {
            term_re = sigs[nperm] * overall_weight * src_weights[wnum] * (B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,n ,Nc_f,Ns_f,Nsnk_f)] * B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,n ,Nc_f,Ns_f,Nsnk_f)] - B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,n ,Nc_f,Ns_f,Nsnk_f)] * B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,n ,Nc_f,Ns_f,Nsnk_f)]);
            term_im = sigs[nperm] * overall_weight * src_weights[wnum] * (B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,n ,Nc_f,Ns_f,Nsnk_f)] * B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,n ,Nc_f,Ns_f,Nsnk_f)] + B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,n ,Nc_f,Ns_f,Nsnk_f)] * B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,n ,Nc_f,Ns_f,Nsnk_f)]);
            for (m=0; m<Nsrc_fHex; m++) {
               C_re[index_3d(m,n,t ,Nsnk_f,Nt_f)] += hex_src_psi_re[index_2d(y,m, Nsrc_fHex)] * term_re - hex_src_psi_im[index_2d(y,m, Nsrc_fHex)] * term_im;
               C_im[index_3d(m,n,t ,Nsnk_f,Nt_f)] += hex_src_psi_re[index_2d(y,m, Nsrc_fHex)] * term_im + hex_src_psi_im[index_2d(y,m, Nsrc_fHex)] * term_re;
            }
         }
      }
   } 
}

void make_hex_correlator(double* C_re,
    double* C_im,
    const double* B1_prop_re, 
    const double* B1_prop_im, 
    const double* B2_prop_re, 
    const double* B2_prop_im, 
    const int* perms, 
    const int* sigs, 
    const int* B1_src_color_weights, 
    const int* B1_src_spin_weights, 
    const double* B1_src_weights, 
    const int* B2_src_color_weights, 
    const int* B2_src_spin_weights, 
    const double* B2_src_weights, 
    const double overall_weight, 
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
    const double* hex_src_psi_re,
    const double* hex_src_psi_im,
    const double* hex_snk_psi_re,
    const double* hex_snk_psi_im,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw2Hex_f,
    const int Nq_f,
    const int Nsrc_fHex,
    const int Nsnk_fHex,
    const int Nperms_f) {
   /* indices */
   int x,t,wnum,nperm,b,n,m,y,wnumprime;
   int iC1prime,iS1prime,jC1prime,jS1prime,kC1prime,kS1prime,iC2prime,iS2prime,jC2prime,jS2prime,kC2prime,kS2prime;
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2;
   int Nb_f = 2;
   std::complex<double> B1_prop_prod_02, B1_prop_prod, B2_prop_prod_02, B2_prop_prod;
   std::complex<double> prop_prod, new_prop_prod;
   /* build dibaryon */
   int snk_1_nq[Nb_f];
   int snk_2_nq[Nb_f];
   int snk_3_nq[Nb_f];
   int snk_1_b[Nb_f];
   int snk_2_b[Nb_f];
   int snk_3_b[Nb_f];
   int snk_1[Nb_f];
   int snk_2[Nb_f];
   int snk_3[Nb_f];
   for (nperm=0; nperm<Nperms_f; nperm++) {
      for (b=0; b<Nb_f; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq_f*b+0 ,2*Nq_f)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq_f*b+1 ,2*Nq_f)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq_f*b+2 ,2*Nq_f)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq_f) / Nq_f;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq_f) / Nq_f;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq_f) / Nq_f;
         snk_1_nq[b] = snk_1[b] % Nq_f;
         snk_2_nq[b] = snk_2[b] % Nq_f;
         snk_3_nq[b] = snk_3[b] % Nq_f;
      }
      for (wnumprime=0; wnumprime< Nw2Hex_f; wnumprime++) {
         iC1prime = snk_color_weights[index_3d(snk_1_b[0],wnumprime,snk_1_nq[0] ,Nw2Hex_f,Nq_f)];
         iS1prime = snk_spin_weights[index_3d(snk_1_b[0],wnumprime,snk_1_nq[0] ,Nw2Hex_f,Nq_f)];
         jC1prime = snk_color_weights[index_3d(snk_2_b[0],wnumprime,snk_2_nq[0] ,Nw2Hex_f,Nq_f)];
         jS1prime = snk_spin_weights[index_3d(snk_2_b[0],wnumprime,snk_2_nq[0] ,Nw2Hex_f,Nq_f)];
         kC1prime = snk_color_weights[index_3d(snk_3_b[0],wnumprime,snk_3_nq[0] ,Nw2Hex_f,Nq_f)];
         kS1prime = snk_spin_weights[index_3d(snk_3_b[0],wnumprime,snk_3_nq[0] ,Nw2Hex_f,Nq_f)];
         iC2prime = snk_color_weights[index_3d(snk_1_b[1],wnumprime,snk_1_nq[1] ,Nw2Hex_f,Nq_f)];
         iS2prime = snk_spin_weights[index_3d(snk_1_b[1],wnumprime,snk_1_nq[1] ,Nw2Hex_f,Nq_f)];
         jC2prime = snk_color_weights[index_3d(snk_2_b[1],wnumprime,snk_2_nq[1] ,Nw2Hex_f,Nq_f)];
         jS2prime = snk_spin_weights[index_3d(snk_2_b[1],wnumprime,snk_2_nq[1] ,Nw2Hex_f,Nq_f)];
         kC2prime = snk_color_weights[index_3d(snk_3_b[1],wnumprime,snk_3_nq[1] ,Nw2Hex_f,Nq_f)];
         kS2prime = snk_spin_weights[index_3d(snk_3_b[1],wnumprime,snk_3_nq[1] ,Nw2Hex_f,Nq_f)]; 
         for (t=0; t<Nt_f; t++) {
            for (y=0; y<Vsrc_f; y++) {
               for (x=0; x<Vsnk_f; x++) {
                  std::complex<double> B1_prop_prod_re(0, 0);
                  for (wnum=0; wnum<Nw2Hex_f; wnum++) {
                     iC1 = snk_color_weights[index_3d(0,wnum,0 ,Nw2Hex_f,Nq_f)];
                     iS1 = snk_spin_weights[index_3d(0,wnum,0 ,Nw2Hex_f,Nq_f)];
                     jC1 = snk_color_weights[index_3d(0,wnum,1 ,Nw2Hex_f,Nq_f)];
                     jS1 = snk_spin_weights[index_3d(0,wnum,1 ,Nw2Hex_f,Nq_f)];
                     kC1 = snk_color_weights[index_3d(0,wnum,2 ,Nw2Hex_f,Nq_f)];
                     kS1 = snk_spin_weights[index_3d(0,wnum,2 ,Nw2Hex_f,Nq_f)];
                     iC2 = snk_color_weights[index_3d(1,wnum,0 ,Nw2Hex_f,Nq_f)];
                     iS2 = snk_spin_weights[index_3d(1,wnum,0 ,Nw2Hex_f,Nq_f)];
                     jC2 = snk_color_weights[index_3d(1,wnum,1 ,Nw2Hex_f,Nq_f)];
                     jS2 = snk_spin_weights[index_3d(1,wnum,1 ,Nw2Hex_f,Nq_f)];
                     kC2 = snk_color_weights[index_3d(1,wnum,2 ,Nw2Hex_f,Nq_f)];
                     kS2 = snk_spin_weights[index_3d(1,wnum,2 ,Nw2Hex_f,Nq_f)]; 
                     std::complex<double> B1_prop_0(B1_prop_re[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B1_prop_im[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B1_prop_2(B1_prop_re[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B1_prop_im[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B1_prop_1(B1_prop_re[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B1_prop_im[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     B1_prop_prod = B1_prop_0 * B1_prop_1 * B1_prop_2;
                     std::complex<double> B2_prop_0(B2_prop_re[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B2_prop_im[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B2_prop_2(B2_prop_re[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B2_prop_im[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B2_prop_1(B2_prop_re[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B2_prop_im[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     B2_prop_prod = B2_prop_0 * B2_prop_1 * B2_prop_2;
                     prop_prod = overall_weight * sigs[nperm] * snk_weights[wnumprime] * snk_weights[wnum] * B1_prop_prod * B2_prop_prod;
                     for (m=0; m<Nsrc_fHex; m++) {
                        std::complex<double> src_psi(hex_src_psi_re[index_2d(y,m ,Nsrc_fHex)],  hex_src_psi_im[index_2d(y,m ,Nsrc_fHex)]);
                        new_prop_prod = prop_prod * src_psi;
                        for (n=0; n<Nsnk_fHex; n++) {
                           std::complex<double> snk_psi(hex_snk_psi_re[index_2d(x,n ,Nsnk_fHex)], hex_snk_psi_im[index_2d(x,n ,Nsnk_fHex)]);
                           std::complex<double> corr = new_prop_prod * snk_psi;
                           C_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] += real(corr);
                           C_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] += imag(corr);
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void make_two_nucleon_2pt(double* C_re,
    double* C_im,
    const double* B1_prop_re, 
    const double* B1_prop_im, 
    const double* B2_prop_re, 
    const double* B2_prop_im, 
    const int* src_color_weights_r1, 
    const int* src_spin_weights_r1, 
    const double* src_weights_r1, 
    const int* src_color_weights_r2, 
    const int* src_spin_weights_r2, 
    const double* src_weights_r2, 
    const int *hex_snk_color_weights_A1,
    const int *hex_snk_spin_weights_A1,
    const double *hex_snk_weights_A1,
    const int *hex_snk_color_weights_T1_r1,
    const int *hex_snk_spin_weights_T1_r1,
    const double *hex_snk_weights_T1_r1,
    const int *hex_snk_color_weights_T1_r2,
    const int *hex_snk_spin_weights_T1_r2,
    const double *hex_snk_weights_T1_r2,
    const int *hex_snk_color_weights_T1_r3,
    const int *hex_snk_spin_weights_T1_r3,
    const double *hex_snk_weights_T1_r3,
    const int* perms, 
    const int* sigs, 
    const double* src_psi_B1_re, 
    const double* src_psi_B1_im, 
    const double* src_psi_B2_re, 
    const double* src_psi_B2_im, 
    const double* snk_psi_re,
    const double* snk_psi_im,
    const double* snk_psi_B1_re, 
    const double* snk_psi_B1_im, 
    const double* snk_psi_B2_re, 
    const double* snk_psi_B2_im, 
    const double* hex_src_psi_re, 
    const double* hex_src_psi_im, 
    const double* hex_snk_psi_re, 
    const double* hex_snk_psi_im,
    const int space_symmetric,
    const int snk_entangled,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nw2Hex_f,
    const int Nq_f,
    const int Nsrc_f,
    const int Nsnk_f,
    const int Nsrc_fHex,
    const int Nsnk_fHex,
    const int Nperms_f) {
      total_time -= clock();
      printf("Running reference code with Vsrc = %d, Nw = %d, Nsrc = %d\n", Vsrc_f, Nw_f, Nsrc_f);
   /* indices */
   double overall_weight = 1.0/2.0;
   int nB1, nB2, nq, n, m, t, x1, x2, x, y;
   // hold results for two nucleon correlators 
   double* BB_0_re = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_0_im = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r1_re = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r1_im = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r2_re = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r2_im = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r3_re = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r3_im = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   for (m=0; m<Nsrc_f; m++) {
      for (n=0; n<Nsnk_f; n++) {
         for (t=0; t<Nt_f; t++) {
            BB_0_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_0_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r1_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r1_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r2_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r2_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r3_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r3_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
         }
      }
   }
   double* BB_H_0_re = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_0_im = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r1_re = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r1_im = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r2_re = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r2_im = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r3_re = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r3_im = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   for (m=0; m<Nsrc_f; m++) {
      for (n=0; n<Nsnk_fHex; n++) {
         for (t=0; t<Nt_f; t++) {
            BB_H_0_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_0_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r1_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r1_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r2_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r2_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r3_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r3_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
         }
      }
   }
   double* H_BB_0_re = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_0_im = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r1_re = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r1_im = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r2_re = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r2_im = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r3_re = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r3_im = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   for (m=0; m<Nsrc_fHex; m++) {
      for (n=0; n<Nsnk_f; n++) {
         for (t=0; t<Nt_f; t++) {
            H_BB_0_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_0_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r1_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r1_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r2_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r2_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r3_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r3_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0; 
         }
      }
   }
   double* H_0_re = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_0_im = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r1_re = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r1_im = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r2_re = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r2_im = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r3_re = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r3_im = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   for (m=0; m<Nsrc_fHex; m++) {
      for (n=0; n<Nsnk_fHex; n++) {
         for (t=0; t<Nt_f; t++) {
            H_0_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_0_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r1_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r1_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r2_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r2_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r3_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r3_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
         }
      }
   }
   /* compute two nucleon snk weights */
   int Nw2_f = Nw_f*Nw_f;
   int* snk_color_weights_1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_2 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_r1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_r2_1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_r2_2 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_r3 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_2 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_r1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_r2_1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_r2_2 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_r3 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   double* snk_weights_1 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_2 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_r1 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_r2_1 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_r2_2 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_r3 = (double *) malloc(Nw2_f * sizeof (double));
   for (nB1=0; nB1<Nw_f; nB1++) {
      for (nB2=0; nB2<Nw_f; nB2++) {
         snk_weights_1[nB1+Nw_f*nB2] = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         snk_weights_2[nB1+Nw_f*nB2] = -1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         snk_weights_r1[nB1+Nw_f*nB2] = src_weights_r1[nB1]*src_weights_r1[nB2];
         snk_weights_r2_1[nB1+Nw_f*nB2] = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         snk_weights_r2_2[nB1+Nw_f*nB2] = 1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         snk_weights_r3[nB1+Nw_f*nB2] = src_weights_r2[nB1]*src_weights_r2[nB2];
         for (nq=0; nq<Nq_f; nq++) {
            // A1g
            snk_color_weights_1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_color_weights_2[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_2[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_2[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_2[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq_f)];
            // T1g_r1
            snk_color_weights_r1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_r1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_r1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_r1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq_f)];
            // T1g_r2
            snk_color_weights_r2_1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_r2_1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_r2_1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_r2_1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_color_weights_r2_2[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_r2_2[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_r2_2[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_r2_2[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq_f)];
            // T1g_r3 
            snk_color_weights_r3[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_r3[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_r3[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_r3[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq_f)];
         }
      }
   }
   int* hex_snk_color_weights_0 = (int *) malloc(2 * Nw2Hex_f * Nq_f * sizeof (int));
   int* hex_snk_color_weights_r1 = (int *) malloc(2 * Nw2Hex_f * Nq_f * sizeof (int));
   int* hex_snk_color_weights_r2 = (int *) malloc(2 * Nw2Hex_f * Nq_f * sizeof (int));
   int* hex_snk_color_weights_r3 = (int *) malloc(2 * Nw2Hex_f * Nq_f * sizeof (int));
   int* hex_snk_spin_weights_0 = (int *) malloc(2 * Nw2Hex_f * Nq_f * sizeof (int));
   int* hex_snk_spin_weights_r1 = (int *) malloc(2 * Nw2Hex_f * Nq_f * sizeof (int));
   int* hex_snk_spin_weights_r2 = (int *) malloc(2 * Nw2Hex_f * Nq_f * sizeof (int));
   int* hex_snk_spin_weights_r3 = (int *) malloc(2 * Nw2Hex_f * Nq_f * sizeof (int));
   double* hex_snk_weights_0 = (double *) malloc(Nw2Hex_f * sizeof (double));
   double* hex_snk_weights_r1 = (double *) malloc(Nw2Hex_f * sizeof (double));
   double* hex_snk_weights_r2 = (double *) malloc(Nw2Hex_f * sizeof (double));
   double* hex_snk_weights_r3 = (double *) malloc(Nw2Hex_f * sizeof (double));
   for (int b=0; b< 2; b++) {
      for (int wnum=0; wnum< Nw2Hex_f; wnum++) {
         hex_snk_weights_0[wnum] = hex_snk_weights_A1[wnum];
         hex_snk_weights_r1[wnum] = hex_snk_weights_T1_r1[wnum];
         hex_snk_weights_r2[wnum] = hex_snk_weights_T1_r2[wnum];
         hex_snk_weights_r3[wnum] = hex_snk_weights_T1_r3[wnum];
         for (int q=0; q < Nq; q++) {
            hex_snk_color_weights_0[index_3d(b, wnum, q, Nw2Hex_f, Nq)] = hex_snk_color_weights_A1[index_2d(wnum,b*Nq+q ,2*Nq)];
            hex_snk_spin_weights_0[index_3d(b, wnum, q, Nw2Hex_f, Nq)] = hex_snk_spin_weights_A1[index_2d(wnum,b*Nq+q ,2*Nq)];
            hex_snk_color_weights_r1[index_3d(b, wnum, q, Nw2Hex_f, Nq)] = hex_snk_color_weights_T1_r1[index_2d(wnum,b*Nq+q ,2*Nq)];
            hex_snk_spin_weights_r1[index_3d(b, wnum, q, Nw2Hex_f, Nq)] = hex_snk_spin_weights_T1_r1[index_2d(wnum,b*Nq+q ,2*Nq)];
            hex_snk_color_weights_r2[index_3d(b, wnum, q, Nw2Hex_f, Nq)] = hex_snk_color_weights_T1_r2[index_2d(wnum,b*Nq+q ,2*Nq)];
            hex_snk_spin_weights_r2[index_3d(b, wnum, q, Nw2Hex_f, Nq)] = hex_snk_spin_weights_T1_r2[index_2d(wnum,b*Nq+q ,2*Nq)];
            hex_snk_color_weights_r3[index_3d(b, wnum, q, Nw2Hex_f, Nq)] = hex_snk_color_weights_T1_r3[index_2d(wnum,b*Nq+q ,2*Nq)];
            hex_snk_spin_weights_r3[index_3d(b, wnum, q, Nw2Hex_f, Nq)] = hex_snk_spin_weights_T1_r3[index_2d(wnum,b*Nq+q ,2*Nq)];
         }
      }
   }
   
   printf("made snk weights \n");
   if (Nsrc_f > 0 && Nsnk_f > 0) {
         /* BB_BB */
      int block_size = Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f;
      double* B1_Blocal_r1_re_all = (double *) malloc(Vsnk_f * block_size * sizeof (double));
      double* B1_Blocal_r1_im_all = (double *) malloc(Vsnk_f * block_size * sizeof (double));
      double* B1_Blocal_r2_re_all = (double *) malloc(Vsnk_f * block_size * sizeof (double));
      double* B1_Blocal_r2_im_all = (double *) malloc(Vsnk_f * block_size * sizeof (double));
      double* B2_Blocal_r1_re_all = (double *) malloc(Vsnk_f * block_size * sizeof (double));
      double* B2_Blocal_r1_im_all = (double *) malloc(Vsnk_f * block_size * sizeof (double));
      double* B2_Blocal_r2_re_all = (double *) malloc(Vsnk_f * block_size * sizeof (double));
      double* B2_Blocal_r2_im_all = (double *) malloc(Vsnk_f * block_size * sizeof (double));
      double* B1_Blocal_r1_re;
      double* B1_Blocal_r1_im;
      double* B1_Blocal_r2_re;
      double* B1_Blocal_r2_im;
      double* B2_Blocal_r1_re;
      double* B2_Blocal_r1_im;
      double* B2_Blocal_r2_re;
      double* B2_Blocal_r2_im;

      double* B1_Bfirst_r1_re  = (double *) malloc(block_size * sizeof (double));
      double* B1_Bfirst_r1_im  = (double *) malloc(block_size * sizeof (double));
      double* B1_Bfirst_r2_re  = (double *) malloc(block_size * sizeof (double));
      double* B1_Bfirst_r2_im  = (double *) malloc(block_size * sizeof (double));
      double* B2_Bfirst_r1_re  = (double *) malloc(block_size * sizeof (double));
      double* B2_Bfirst_r1_im  = (double *) malloc(block_size * sizeof (double));
      double* B2_Bfirst_r2_re  = (double *) malloc(block_size * sizeof (double));
      double* B2_Bfirst_r2_im  = (double *) malloc(block_size * sizeof (double));
      double* B1_Bsecond_r1_re = (double *) malloc(block_size * sizeof (double));
      double* B1_Bsecond_r1_im = (double *) malloc(block_size * sizeof (double));
      double* B1_Bsecond_r2_re = (double *) malloc(block_size * sizeof (double));
      double* B1_Bsecond_r2_im = (double *) malloc(block_size * sizeof (double));
      double* B2_Bsecond_r1_re = (double *) malloc(block_size * sizeof (double));
      double* B2_Bsecond_r1_im = (double *) malloc(block_size * sizeof (double));
      double* B2_Bsecond_r2_re = (double *) malloc(block_size * sizeof (double));
      double* B2_Bsecond_r2_im = (double *) malloc(block_size * sizeof (double));
      double* B1_Bthird_r1_re  = (double *) malloc(block_size * sizeof (double));
      double* B1_Bthird_r1_im  = (double *) malloc(block_size * sizeof (double));
      double* B1_Bthird_r2_re  = (double *) malloc(block_size * sizeof (double));
      double* B1_Bthird_r2_im  = (double *) malloc(block_size * sizeof (double));
      double* B2_Bthird_r1_re  = (double *) malloc(block_size * sizeof (double));
      double* B2_Bthird_r1_im  = (double *) malloc(block_size * sizeof (double));
      double* B2_Bthird_r2_re  = (double *) malloc(block_size * sizeof (double));
      double* B2_Bthird_r2_im  = (double *) malloc(block_size * sizeof (double));
      for (t=0; t<Nt_f; t++) {
         // precompute local blocks
         // NB: These are equal here but are not equal if src_weights is different for B1 and B2
         for (x =0; x<Vsnk_f; x++) {
            B1_Blocal_r1_re = B1_Blocal_r1_re_all + x * block_size;
            B1_Blocal_r1_im = B1_Blocal_r1_im_all + x * block_size;
            B1_Blocal_r2_re = B1_Blocal_r2_re_all + x * block_size;
            B1_Blocal_r2_im = B1_Blocal_r2_im_all + x * block_size;
            B2_Blocal_r1_re = B2_Blocal_r1_re_all + x * block_size;
            B2_Blocal_r1_im = B2_Blocal_r1_im_all + x * block_size;
            B2_Blocal_r2_re = B2_Blocal_r2_re_all + x * block_size;
            B2_Blocal_r2_im = B2_Blocal_r2_im_all + x * block_size;
            make_local_block(B1_Blocal_r1_re, B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B1_Blocal_r2_re, B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B2_Blocal_r1_re, B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B2_Blocal_r2_re, B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
         }
         for (x1 =0; x1<Vsnk_f; x1++) {
            for (x2 =0; x2<Vsnk_f; x2++) {
               // retrieve local blocks
               B1_Blocal_r1_re = B1_Blocal_r1_re_all + x1 * block_size;
               B1_Blocal_r1_im = B1_Blocal_r1_im_all + x1 * block_size;
               B1_Blocal_r2_re = B1_Blocal_r2_re_all + x1 * block_size;
               B1_Blocal_r2_im = B1_Blocal_r2_im_all + x1 * block_size;
               B2_Blocal_r1_re = B2_Blocal_r1_re_all + x2 * block_size;
               B2_Blocal_r1_im = B2_Blocal_r1_im_all + x2 * block_size;
               B2_Blocal_r2_re = B2_Blocal_r2_re_all + x2 * block_size;
               B2_Blocal_r2_im = B2_Blocal_r2_im_all + x2 * block_size;
               // create blocks
               make_first_block(B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_first_block(B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_second_block(B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_second_block(B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_third_block(B1_Bthird_r1_re, B1_Bthird_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_third_block(B1_Bthird_r2_re, B1_Bthird_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);

               make_first_block(B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_first_block(B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_second_block(B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_second_block(B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_third_block(B2_Bthird_r1_re, B2_Bthird_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_third_block(B2_Bthird_r2_re, B2_Bthird_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               /* compute two nucleon correlators from blocks */
               int* src_spins = (int *) malloc(2 * sizeof (int));
               src_spins[0] = 1;
               src_spins[1] = 2;
               make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 2;
               src_spins[1] = 1;
               make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 1;
               src_spins[1] = 1;
               make_dibaryon_correlator(BB_r1_re, BB_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 1;
               src_spins[1] = 2;
               make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, t, x1, x2,  Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 2;
               src_spins[1] = 1;
               make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 2;
               src_spins[1] = 2;
               make_dibaryon_correlator(BB_r3_re, BB_r3_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, src_spins, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               #if 0
               if (x1 != x2) {
               make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 2;
               src_spins[1] = 1;
               make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 1;
               src_spins[1] = 1;
               make_dibaryon_correlator(BB_r1_re, BB_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 1;
               src_spins[1] = 2;
               make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, t, x2, x1,  Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 2;
               src_spins[1] = 1;
               make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               src_spins[0] = 2;
               src_spins[1] = 2;
               make_dibaryon_correlator(BB_r3_re, BB_r3_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bfirst_r1_re, B2_Bfirst_r1_im, B2_Bsecond_r1_re, B2_Bsecond_r1_im, B2_Bthird_r1_re, B2_Bthird_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bfirst_r1_re, B1_Bfirst_r1_im, B1_Bsecond_r1_re, B1_Bsecond_r1_im, B1_Bthird_r1_re, B1_Bthird_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bfirst_r2_re, B2_Bfirst_r2_im, B2_Bsecond_r2_re, B2_Bsecond_r2_im, B2_Bthird_r2_re, B2_Bthird_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bfirst_r2_re, B1_Bfirst_r2_im, B1_Bsecond_r2_re, B1_Bsecond_r2_im, B1_Bthird_r2_re, B1_Bthird_r2_im, src_spins, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, snk_psi_re, snk_psi_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               }
               #endif
            }
         }
         for (m=0; m<Nsrc_f; m++) {
            for (n=0; n<Nsnk_f; n++) {
               C_re[index_4d(0,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_0_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(0,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_0_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(1,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_r1_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(1,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_r1_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(2,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_r2_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(2,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_r2_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(3,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_r3_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(3,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_r3_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
            }
         }
      }
      free(B1_Blocal_r1_re_all);
      free(B1_Blocal_r1_im_all);
      free(B1_Blocal_r2_re_all);
      free(B1_Blocal_r2_im_all);
      free(B2_Blocal_r1_re_all);
      free(B2_Blocal_r1_im_all);
      free(B2_Blocal_r2_re_all);
      free(B2_Blocal_r2_im_all);

      free(B1_Bfirst_r1_re);
      free(B1_Bfirst_r1_im);
      free(B1_Bfirst_r2_re);
      free(B1_Bfirst_r2_im);
      free(B1_Bsecond_r1_re);
      free(B1_Bsecond_r1_im);
      free(B1_Bsecond_r2_re);
      free(B1_Bsecond_r2_im);
      free(B1_Bthird_r1_re);
      free(B1_Bthird_r1_im);
      free(B1_Bthird_r2_re);
      free(B1_Bthird_r2_im);
      free(B2_Bfirst_r1_re);
      free(B2_Bfirst_r1_im);
      free(B2_Bfirst_r2_re);
      free(B2_Bfirst_r2_im);
      free(B2_Bsecond_r1_re);
      free(B2_Bsecond_r1_im);
      free(B2_Bsecond_r2_re);
      free(B2_Bsecond_r2_im);
      free(B2_Bthird_r1_re);
      free(B2_Bthird_r1_im);
      free(B2_Bthird_r2_re);
      free(B2_Bthird_r2_im);
      printf("made BB-BB \n");
   }
   free(BB_0_re);
   free(BB_0_im);
   free(BB_r1_re);
   free(BB_r1_im);
   free(BB_r2_re);
   free(BB_r2_im);
   free(BB_r3_re);
   free(BB_r3_im);
   if (Nsrc_f > 0 && Nsnk_fHex > 0) {
      // BB_H 
      double* B1_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      for (t=0; t<Nt_f; t++) {
         for (x =0; x<Vsnk_f; x++) {
            // create blocks
            make_local_block(B1_Blocal_r1_re, B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B1_Blocal_r2_re, B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B2_Blocal_r1_re, B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B2_Blocal_r2_re, B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), hex_snk_color_weights_0, hex_snk_spin_weights_0, hex_snk_weights_0, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), hex_snk_color_weights_0, hex_snk_spin_weights_0, hex_snk_weights_0, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r1_re, BB_H_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight, hex_snk_color_weights_r1, hex_snk_spin_weights_r1, hex_snk_weights_r1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), hex_snk_color_weights_r2, hex_snk_spin_weights_r2, hex_snk_weights_r2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), hex_snk_color_weights_r2, hex_snk_spin_weights_r2, hex_snk_weights_r2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r3_re, BB_H_r3_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight, hex_snk_color_weights_r3, hex_snk_spin_weights_r3, hex_snk_weights_r3, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
         }
         for (m=0; m<Nsrc_f; m++) {
            for (n=0; n<Nsnk_fHex; n++) {
               C_re[index_4d(0,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_H_0_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(0,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_H_0_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(1,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_H_r1_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(1,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_H_r1_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(2,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_H_r2_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(2,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_H_r2_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(3,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_H_r3_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(3,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += BB_H_r3_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
            }
         } 
      }
      free(B1_Blocal_r1_re);
      free(B1_Blocal_r1_im);
      free(B1_Blocal_r2_re);
      free(B1_Blocal_r2_im);
      free(B2_Blocal_r1_re);
      free(B2_Blocal_r1_im);
      free(B2_Blocal_r2_re);
      free(B2_Blocal_r2_im);
      printf("made BB-H \n");
   }
   free(BB_H_0_re);
   free(BB_H_0_im);
   free(BB_H_r1_re);
   free(BB_H_r1_im);
   free(BB_H_r2_re);
   free(BB_H_r2_im);
   free(BB_H_r3_re);
   free(BB_H_r3_im);
   if (Nsrc_fHex > 0 && Nsnk_fHex > 0) {
      // H_H 
      make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight, hex_snk_color_weights_0, hex_snk_spin_weights_0, hex_snk_weights_0, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r1_re, H_r1_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight, hex_snk_color_weights_r1, hex_snk_spin_weights_r1, hex_snk_weights_r1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight, hex_snk_color_weights_r2, hex_snk_spin_weights_r2, hex_snk_weights_r2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r3_re, H_r3_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight, hex_snk_color_weights_r3, hex_snk_spin_weights_r3, hex_snk_weights_r3, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      for (m=0; m<Nsrc_fHex; m++) {
         for (n=0; n<Nsnk_fHex; n++) {
            for (t=0; t<Nt_f; t++) {
               C_re[index_4d(0,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_0_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(0,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_0_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(1,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_r1_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(1,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_r1_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(2,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_r2_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(2,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_r2_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(3,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_r3_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(3,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_r3_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
            }
         }
      }
      printf("made H-H \n");
   }
   free(H_0_re);
   free(H_0_im);
   free(H_r1_re);
   free(H_r1_im);
   free(H_r2_re);
   free(H_r2_im);
   free(H_r3_re);
   free(H_r3_im);
   if (Nsnk_f > 0 && Nsrc_fHex > 0 && snk_entangled == 0) {
      // H_BB 
      double* snk_B1_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B1_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B1_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B1_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B2_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B2_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B2_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B2_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      for (t=0; t<Nt_f; t++) {
         for (y =0; y<Vsrc_f; y++) {
            // create blocks
            make_local_snk_block(snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, snk_psi_B1_re, snk_psi_B1_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsnk_f);
            make_local_snk_block(snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_psi_B1_re, snk_psi_B1_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsnk_f);
            make_local_snk_block(snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, snk_psi_B2_re, snk_psi_B2_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsnk_f);
            make_local_snk_block(snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_psi_B2_re, snk_psi_B2_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsnk_f);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2), hex_snk_color_weights_0, hex_snk_spin_weights_0, hex_snk_weights_0, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2), hex_snk_color_weights_0, hex_snk_spin_weights_0, hex_snk_weights_0, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r1_re, H_BB_r1_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight, hex_snk_color_weights_r1, hex_snk_spin_weights_r1, hex_snk_weights_r1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2), hex_snk_color_weights_r2, hex_snk_spin_weights_r2, hex_snk_weights_r2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2), hex_snk_color_weights_r2, hex_snk_spin_weights_r2, hex_snk_weights_r2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r3_re, H_BB_r3_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight, hex_snk_color_weights_r3, hex_snk_spin_weights_r3, hex_snk_weights_r3, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw2Hex_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
         }
         for (m=0; m<Nsrc_fHex; m++) {
            for (n=0; n<Nsnk_f; n++) {
               C_re[index_4d(0,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_BB_0_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(0,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_BB_0_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(1,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_BB_r1_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(1,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_BB_r1_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(2,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_BB_r2_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(2,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_BB_r2_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(3,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_BB_r3_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(3,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] += H_BB_r3_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
            }
         }
      }
      free(snk_B1_Blocal_r1_re);
      free(snk_B1_Blocal_r1_im);
      free(snk_B1_Blocal_r2_re);
      free(snk_B1_Blocal_r2_im);
      free(snk_B2_Blocal_r1_re);
      free(snk_B2_Blocal_r1_im);
      free(snk_B2_Blocal_r2_re);
      free(snk_B2_Blocal_r2_im);
      printf("made H-BB \n");
   }
   free(H_BB_0_re);
   free(H_BB_0_im);
   free(H_BB_r1_re);
   free(H_BB_r1_im);
   free(H_BB_r2_re);
   free(H_BB_r2_im);
   free(H_BB_r3_re);
   free(H_BB_r3_im);
   total_time += clock();
   printf("Time in make_local_block: %f\n", ((float) local_block_time) / CLOCKS_PER_SEC);
   printf("Time in make_first_block: %f\n", ((float) first_block_time) / CLOCKS_PER_SEC);
   printf("Time in make_second_block: %f\n", ((float) second_block_time) / CLOCKS_PER_SEC);
   printf("Time in make_third_block: %f\n", ((float) third_block_time) / CLOCKS_PER_SEC);
   printf("Time in make_dibaryon_correlator: %f\n", ((float) correlator_time) / CLOCKS_PER_SEC);
   printf("Total time: %f\n", ((float) total_time) / CLOCKS_PER_SEC);
}

