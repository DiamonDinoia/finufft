c     Fixed-form Fortran consumer of the installed finufft.fh header.
      program app
      implicit none
      include 'finufft.fh'

      integer ier,iflag
      integer*8 M,N,j,k,kidx
      real*8 tol,pi,err,fmax
      parameter (pi=3.141592653589793d0)
      real*8, allocatable :: xj(:)
      complex*16, allocatable :: cj(:),fk(:)
      complex*16 fktest
      type(finufft_opts), pointer :: defopts => null()

      M = 100000
      N = 1000
      allocate(xj(M))
      allocate(cj(M))
      allocate(fk(N))
      do j = 1,M
         xj(j) = pi * dcos(pi*j/M)
         cj(j) = dcmplx(dsin((100d0*j)/M), dcos(1d0+(50d0*j)/M))
      enddo

      iflag = 1
      tol = 1d-9
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,defopts,ier)
      if (ier.gt.1) then
         print '("finufft1d1 failed, ier=",i0)', ier
         stop 1
      endif

c     compare one mode against the direct sum
      k = N/3
      kidx = k + N/2 + 1
      fktest = dcmplx(0d0,0d0)
      do j = 1,M
         fktest = fktest + cj(j)*dcmplx(dcos(k*xj(j)),dsin(k*xj(j)))
      enddo
      fmax = 0d0
      do j = 1,N
         fmax = max(fmax,abs(fk(j)))
      enddo
      err = abs(fk(kidx)-fktest)/fmax
      print '("finufft1d1: ier=",i0,", rel err in mode ",i0," is ",
     $     e9.2)', ier, k, err
      if (err.ge.1d-7) stop 1

      deallocate(xj,cj,fk)
      end program app
