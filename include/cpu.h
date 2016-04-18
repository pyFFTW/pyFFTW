/*
 * Copyright 2016 Knowledge Economy Developments Ltd
 *
 * Henry Gomersall
 * heng@kedevelopments.co.uk
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/* Small utilities for inspecting the CPU */

#ifndef CPU_H
#define CPU_H

#if __STDC_VERSION__ >= 199901L
  /* "inline" is a keyword */
#else
# define inline
#endif

#if defined(__amd64__) || defined (_M_X64) || defined(__i386__) || defined(_M_IX86) || defined(_X86_)

  #define AVX_WORD 2
  #define AVX_BIT 28
  #define SSE_WORD 3
  #define SSE_BIT 25

  #ifdef _MSC_VER
    /* Visual Studio Code */
    #include <intrin.h>
    #define cpuid(func, cpuinfo)\
      __cpuid(cpuinfo, func);

  #else
    /* generic x86 Assembly code (based on wikipedia example)
     * Firstly it's necessary to move ebx into an interim
     * register to protect it (cpuid clobbers eax, ebx ecx and edx)
     * */
    #define cpuid(func, cpuinfo)\
      cpuinfo[0] = func; /* Load the first entry with the func id */\
      __asm__ __volatile__ \
      ("mov %%ebx, %%edi;" /* 32bit PIC: don't clobber ebx */ \
       "cpuid;" \
       "mov %%ebx, %%esi;" \
       "mov %%edi, %%ebx;" \
       :"+a" (cpuinfo[0]), "=S" (cpuinfo[1]), /* eax rw, esi read */ \
       "=c" (cpuinfo[2]), "=d" (cpuinfo[3]) /* ecx read, edx read */\
       : :"edi")

  #endif

/* Returns the byte alignment for optimum simd operations */
static inline int simd_alignment(void){
    int cpuinfo[4];

    /* This gets the cpuinfo (set by 1)*/
    cpuid(1, cpuinfo);

    if (cpuinfo[AVX_WORD] & (1<<AVX_BIT))  /* AVX */
        return 32;
    else if (cpuinfo[SSE_WORD] & (1<<SSE_BIT))  /* SSE */
        return 16;
    else  /* No SIMD */
        return 4;
}

#else

static inline int simd_alignment(void){
    return 4;
}
#endif

#endif /* Header guard */
