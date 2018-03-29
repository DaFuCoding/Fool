// This is the exact cblas.h header file, placed here purely in order to get
// the enums.

#ifndef CBLAS_ENUM_DEFINED_H
   #define CBLAS_ENUM_DEFINED_H
   enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
   enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
                         AtlasConj=114};
   enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
   enum CBLAS_DIAG  {CblasNonUnit=131, CblasUnit=132};
   enum CBLAS_SIDE  {CblasLeft=141, CblasRight=142};
#endif