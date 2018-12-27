typedef enum ___LIB_VERSIONIMF_TYPE {
     _IEEE_ = -1    /* IEEE-like behavior    */
    ,_SVID_         /* SysV, Rel. 4 behavior */
    ,_XOPEN_        /* Unix98                */
    ,_POSIX_        /* Posix                 */
    ,_ISOC_         /* ISO C9X               */
} _LIB_VERSIONIMF_TYPE;

#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
struct int2;

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos);

#endif
