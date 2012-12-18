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

#ifdef _MSC_VER
  /* Visual Studio Code */
  #include <intrin.h>
  #define cpuid(func, cpuinfo)\
    __cpuid(cpuinfo, func);
 
#else
  /* Assembly code (based on wikipedia example) 
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
     : :"edi");
#endif

