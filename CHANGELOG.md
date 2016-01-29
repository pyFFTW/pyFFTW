# Change Log

## [v0.10.1](https://github.com/hgomersall/pyFFTW/tree/v0.10.1) (2016-01-29)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.10.0...v0.10.1)

## [v0.10.0](https://github.com/hgomersall/pyFFTW/tree/v0.10.0) (2016-01-29)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.9.2...v0.10.0)

**Closed issues:**

- Conda downloads are failing [\#76](https://github.com/hgomersall/pyFFTW/issues/76)
- Python 3.4 and WinPython [\#74](https://github.com/hgomersall/pyFFTW/issues/74)
- Installing pyfftw on Anaconda3 on Windows 7 [\#73](https://github.com/hgomersall/pyFFTW/issues/73)
- is python 3.5 supported? [\#71](https://github.com/hgomersall/pyFFTW/issues/71)
- deadlock of cache handler at interpreter shutdown [\#69](https://github.com/hgomersall/pyFFTW/issues/69)
- pyFFTW breaks when forked [\#65](https://github.com/hgomersall/pyFFTW/issues/65)
- build with mingw [\#62](https://github.com/hgomersall/pyFFTW/issues/62)
- Striding in n\_byte\_align not on a uniform standard [\#61](https://github.com/hgomersall/pyFFTW/issues/61)
- No exception on wrong arguments of function call of pyfftw.FFTW\(...\) [\#60](https://github.com/hgomersall/pyFFTW/issues/60)
- pyfftw vs numpy.fft: faster and slower [\#58](https://github.com/hgomersall/pyFFTW/issues/58)
- simple transposes? [\#57](https://github.com/hgomersall/pyFFTW/issues/57)
- `Datatype not supported` with scipy [\#56](https://github.com/hgomersall/pyFFTW/issues/56)
- Update tutorial with new byte align functions [\#53](https://github.com/hgomersall/pyFFTW/issues/53)
- OS X Installation errors:  [\#52](https://github.com/hgomersall/pyFFTW/issues/52)
- Wrong results for pyfftw's ifft? [\#51](https://github.com/hgomersall/pyFFTW/issues/51)
- Installing on OS X Mavericks [\#49](https://github.com/hgomersall/pyFFTW/issues/49)
- Install error. Ld cannot find -lfftw3f [\#48](https://github.com/hgomersall/pyFFTW/issues/48)
- new source release with updated licensing  [\#43](https://github.com/hgomersall/pyFFTW/issues/43)
- Crash during initialization of FFTW plan for r2c and c2r with Intel compiler [\#37](https://github.com/hgomersall/pyFFTW/issues/37)
- Move FFTW class definition to pyfftw.pxd [\#36](https://github.com/hgomersall/pyFFTW/issues/36)
- Provide transform metadata such as axes and direction [\#34](https://github.com/hgomersall/pyFFTW/issues/34)
- Provide shape and dtype properties [\#33](https://github.com/hgomersall/pyFFTW/issues/33)
- problem with very large arrays: OverflowError: value too large to convert to int [\#30](https://github.com/hgomersall/pyFFTW/issues/30)
- add support for in-place multidimensional r2c  transform [\#29](https://github.com/hgomersall/pyFFTW/issues/29)
- Add numpy interface for hfft [\#28](https://github.com/hgomersall/pyFFTW/issues/28)
- add cython as a build dependency [\#25](https://github.com/hgomersall/pyFFTW/issues/25)
- Potential memory leak in caching [\#22](https://github.com/hgomersall/pyFFTW/issues/22)
- Allow GIL to be released with threads=1 [\#13](https://github.com/hgomersall/pyFFTW/issues/13)
- Building for 64bit windows [\#12](https://github.com/hgomersall/pyFFTW/issues/12)
- Test failure using numpy 1.6.2  [\#9](https://github.com/hgomersall/pyFFTW/issues/9)
- Remove the requirement for users to specify alignment [\#8](https://github.com/hgomersall/pyFFTW/issues/8)
- pyfftw.interfaces can only handle numpy arrays [\#7](https://github.com/hgomersall/pyFFTW/issues/7)

**Merged pull requests:**

- Release GIL during both single- and multi-thread execution \(cleaned up patch\) [\#81](https://github.com/hgomersall/pyFFTW/pull/81) ([zpincus](https://github.com/zpincus))
- Support FFTW\_WISDOM\_ONLY \(cleaned up patch\) [\#80](https://github.com/hgomersall/pyFFTW/pull/80) ([zpincus](https://github.com/zpincus))
- Release GIL around FFTW planning [\#78](https://github.com/hgomersall/pyFFTW/pull/78) ([zpincus](https://github.com/zpincus))
- Updated the tutorial to reflect changes with new aligned array creation functions. [\#54](https://github.com/hgomersall/pyFFTW/pull/54) ([drwells](https://github.com/drwells))
- Add description for installing on OS X [\#50](https://github.com/hgomersall/pyFFTW/pull/50) ([arve0](https://github.com/arve0))
- close issue \#8 \(More numpy like aligned array creation functions\) [\#44](https://github.com/hgomersall/pyFFTW/pull/44) ([drwells](https://github.com/drwells))
- Discrete sine transform imports [\#42](https://github.com/hgomersall/pyFFTW/pull/42) ([insertinterestingnamehere](https://github.com/insertinterestingnamehere))

## [v0.9.2](https://github.com/hgomersall/pyFFTW/tree/v0.9.2) (2013-09-20)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.9.2_docs...v0.9.2)

## [v0.9.2_docs](https://github.com/hgomersall/pyFFTW/tree/v0.9.2_docs) (2013-09-11)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.9.1_docs...v0.9.2_docs)

## [v0.9.1_docs](https://github.com/hgomersall/pyFFTW/tree/v0.9.1_docs) (2013-09-11)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.9.1...v0.9.1_docs)

## [v0.9.1](https://github.com/hgomersall/pyFFTW/tree/v0.9.1) (2013-09-11)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.9_docs...v0.9.1)

**Closed issues:**

- Overwriting previous input when cache is enabled. [\#23](https://github.com/hgomersall/pyFFTW/issues/23)
- Race condition in cache culling [\#21](https://github.com/hgomersall/pyFFTW/issues/21)
- Memory corruption at exit [\#19](https://github.com/hgomersall/pyFFTW/issues/19)
- In-place transform? [\#18](https://github.com/hgomersall/pyFFTW/issues/18)
- Support for 2.6? [\#17](https://github.com/hgomersall/pyFFTW/issues/17)
- Install fails; can't find library? [\#16](https://github.com/hgomersall/pyFFTW/issues/16)
- Please include cython source code [\#11](https://github.com/hgomersall/pyFFTW/issues/11)
- Make repeated axes act like numpy.fft's repeated axes [\#2](https://github.com/hgomersall/pyFFTW/issues/2)
- Implement numpy.fft API to pyfftw [\#1](https://github.com/hgomersall/pyFFTW/issues/1)

**Merged pull requests:**

- register fftw cleanup with Py\_AtExit [\#20](https://github.com/hgomersall/pyFFTW/pull/20) ([rainwoodman](https://github.com/rainwoodman))

## [v0.9_docs](https://github.com/hgomersall/pyFFTW/tree/v0.9_docs) (2013-02-15)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.9...v0.9_docs)

## [v0.9](https://github.com/hgomersall/pyFFTW/tree/v0.9) (2013-02-15)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.8.2_docs...v0.9)

**Closed issues:**

- Issue getting PyFFTW to import in Python 3 [\#6](https://github.com/hgomersall/pyFFTW/issues/6)
- Some tests fail [\#5](https://github.com/hgomersall/pyFFTW/issues/5)
- n\_byte\_array\(\) and n\_byte\_align\_empty\(\) break for large arrays [\#4](https://github.com/hgomersall/pyFFTW/issues/4)

## [v0.8.2_docs](https://github.com/hgomersall/pyFFTW/tree/v0.8.2_docs) (2012-05-29)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.8.2...v0.8.2_docs)

## [v0.8.2](https://github.com/hgomersall/pyFFTW/tree/v0.8.2) (2012-05-29)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.8.1_docs...v0.8.2)

**Closed issues:**

- export\_wisdom\(\) fails on windows 7 and vista [\#3](https://github.com/hgomersall/pyFFTW/issues/3)

## [v0.8.1_docs](https://github.com/hgomersall/pyFFTW/tree/v0.8.1_docs) (2012-05-20)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.8.1...v0.8.1_docs)

## [v0.8.1](https://github.com/hgomersall/pyFFTW/tree/v0.8.1) (2012-05-20)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.8.0_docs...v0.8.1)

## [v0.8.0_docs](https://github.com/hgomersall/pyFFTW/tree/v0.8.0_docs) (2012-04-08)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.8.0...v0.8.0_docs)

## [v0.8.0](https://github.com/hgomersall/pyFFTW/tree/v0.8.0) (2012-04-08)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.7.0_docs...v0.8.0)

## [v0.7.0_docs](https://github.com/hgomersall/pyFFTW/tree/v0.7.0_docs) (2012-03-04)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.7.0...v0.7.0_docs)

## [v0.7.0](https://github.com/hgomersall/pyFFTW/tree/v0.7.0) (2012-02-29)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.6.1_docs...v0.7.0)

## [v0.6.1_docs](https://github.com/hgomersall/pyFFTW/tree/v0.6.1_docs) (2012-02-26)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.6.1...v0.6.1_docs)

## [v0.6.1](https://github.com/hgomersall/pyFFTW/tree/v0.6.1) (2012-02-26)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.6.0_docs...v0.6.1)

## [v0.6.0_docs](https://github.com/hgomersall/pyFFTW/tree/v0.6.0_docs) (2012-02-06)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.6.0...v0.6.0_docs)

## [v0.6.0](https://github.com/hgomersall/pyFFTW/tree/v0.6.0) (2012-02-06)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.5.1_docs...v0.6.0)

## [v0.5.1_docs](https://github.com/hgomersall/pyFFTW/tree/v0.5.1_docs) (2012-02-05)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.5.1...v0.5.1_docs)

## [v0.5.1](https://github.com/hgomersall/pyFFTW/tree/v0.5.1) (2012-02-04)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.5.0...v0.5.1)

## [v0.5.0](https://github.com/hgomersall/pyFFTW/tree/v0.5.0) (2012-02-01)
[Full Changelog](https://github.com/hgomersall/pyFFTW/compare/v0.5.0_docs...v0.5.0)

## [v0.5.0_docs](https://github.com/hgomersall/pyFFTW/tree/v0.5.0_docs) (2012-02-01)


\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*