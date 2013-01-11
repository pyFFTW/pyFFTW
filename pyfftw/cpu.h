/*
 * Copyright 2012 Knowledge Economy Developments Ltd
 * 
 * Henry Gomersall
 * heng@kedevelopments.co.uk
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* Small utilities for inspecting the CPU */

#if defined(__amd64__) || defined (_M_X64) || defined(__i386__) || defined(_M_IX86) || defined(_X86_)

  #define AVX_BYTE 2
  #define AVX_BIT 28
  #define SSE_BYTE 3
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
inline int simd_alignment(void){
    int cpuinfo[4];

    /* This gets the cpuinfo (set by 1)*/
    cpuid(1, cpuinfo);

    if (cpuinfo[AVX_BYTE] & (1<<AVX_BIT))  /* AVX */
        return 32;
    else if (cpuinfo[SSE_BYTE] & (1<<SSE_BIT))  /* SSE */
        return 16;
    else  /* No SIMD */
        return 4;
}

#else

inline int simd_alignment(void){
    return 4;
}
#endif

