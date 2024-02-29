!  Generate mesh for liftbody  （without leading)
!  Ver 1.0  (come from grid_body3d-1.3.f90) , 2018-5-5
!  Ver 1.1 2018-5-11, 可输出不同类型的网格文件（PLOT3d or OCFD)
!  Ver 1.2  2018-5-21, 流向网格可调整
!  Ver 1.3  2018-11-4, 法向计算域沿周向可调整  （在seta=180度时， 计算域大幅增加， 以适应当地的横流涡）；
!                      V1.3 采用逐片生成网格方法， 避免了使用三维数组， 可在单机上生成大网格
program main
implicit none
integer::  nx, ns, nh, If_half, If_plotwallmesh, i,j,k
real*8,parameter::  PI=3.1415926535897932d0
real*8,parameter:: a1=20.d0, b1=40.d0, c1=20.d0
real*8:: Phi, ss,sk
real*8, allocatable,dimension(:,:) ::  x2d, y2d, z2d , xn, yn, zn, zh, exp_a, exp_b
real*8,allocatable,dimension(:,:):: tmp2d
real*8,allocatable,dimension(:):: x1d, Phi0, Hend
real*8::  hwall, h_begin, h_end1, h_end2, h_end_seta0
real*8::  x_begin, x_end, XL_part1, XL_Part2, XL_Part3
integer:: nx_part1, nx_part3

open(99,file="grid_body.in")
read(99,*)
read(99,*)
read(99,*)  nx,  ns, nh, If_half, If_plotwallmesh
read(99,*)
read(99,*)  x_begin, x_end, nx_part1, nx_part3, XL_part1, XL_Part2, XL_Part3
read(99,*)
read(99,*)  hwall, h_begin,  h_end1, h_end2, h_end_seta0
read(99,*)
close(99)


h_end_seta0=h_end_seta0*Pi/180.d0

!---------------------------------------------

allocate(x2d(nx,ns), y2d(nx,ns), z2d(nx,ns), x1d(nx) , Phi0(ns), Hend(ns) )
allocate(xn(nx,ns),yn(nx,ns),zn(nx,ns), zh(nx,ns), exp_a(nx,ns), exp_b(nx,ns) )
allocate(tmp2d(nx,ns))

! 流向网格分布 x1d(:)
call  comput_x1d(nx, x1d, x_begin, x_end, nx_part1, nx_part3, XL_part1, XL_Part2, XL_Part3 )
print*, "Comput  X1d OK"
! 周向网格分布 phi0(:)
call comput_phi0(ns,Phi0, If_half )
print*, "Comput phi0 OK"

! Comput wall mesh
call wallmesh2d(nx,ns,x1d,Phi0,x2d,y2d,z2d,If_plotwallmesh)
print*, "Comput Mesh2d OK"
!------------------------ wall-normal  ----------------------
call Wall_Normal(nx,ns,x2d,y2d,z2d,xn,yn,zn,Phi0)
print*, "Comput Wall Normal OK"
!------------------------------
call comput_zh(nx,ns,zh,Phi0,h_begin,h_end1,h_end2,h_end_seta0,IF_half)
print*, "Comput zh OK"
!------------------------------

do j=1,ns
    do i=1,nx
        call grid1d_exponential2(nh,zh(i,j),hwall,exp_a(i,j),exp_b(i,j))   ! 拉伸网格，计算参数exp_a, exp_b : sk=exp_a*exp(ss*(exp_b-1))*Zh
    enddo
enddo

print*, "wall normal grid exponential parameter computing OK "
 !------------------------------
print*, "Output 3D mesh,  nx, ny, nz=", nx, ns, nh
open(99,file="OCFD3d-Mesh.dat", form="unformatted")

print*, "write X3d ..."
do k=1,nh
    ss=(k-1.d0)/(nh-1.d0)
    do j=1,ns
        do i=1,nx
            sk=exp_a(i,j)*(exp(ss*exp_b(i,j))-1.d0)*zh(i,j)
            tmp2d(i,j)=x2d(i,j)+sk*xn(i,j)
        enddo
    enddo
    write(99)  tmp2d
enddo

print*, "write Y3d ..."
do k=1,nh
    ss=(k-1.d0)/(nh-1.d0)
    do j=1,ns
        do i=1,nx
            sk=exp_a(i,j)*(exp(ss*exp_b(i,j))-1.d0)*zh(i,j)
            tmp2d(i,j)=y2d(i,j)+sk*yn(i,j)
        enddo
    enddo
    write(99)  tmp2d
enddo

print*, "write Z3d ..."
do k=1,nh
    ss=(k-1.d0)/(nh-1.d0)
    do j=1,ns
        do i=1,nx
            sk=exp_a(i,j)*(exp(ss*exp_b(i,j))-1.d0)*zh(i,j)
            tmp2d(i,j)=z2d(i,j)+sk*zn(i,j)
        enddo
    enddo
    write(99)  tmp2d
enddo

close(99)
 !---------------------------------------------------------------



deallocate (x2d, y2d, z2d , xn, yn, zn, zh, exp_a, exp_b, tmp2d ,x1d, Phi0, Hend)

end program

!----------------------------------------------------------------------------
! 流向网格：  4段式网格：  均匀、过渡、 均匀、 拉伸
subroutine comput_x1d(nx, x1d, x_begin, x_end, nx_part1, nx_part3, XL_part1, XL_Part2, XL_Part3 )
    implicit none
    integer:: nx, i, nx_part1, nx_part3
    real*8:: x1d(nx), x_begin, x_end, XL_part1, XL_Part2, XL_Part3, XL
    XL=x_end-x_begin
    call get_sx_4part(nx, XL, x1d, nx_part1,nx_part3, XL_part1, XL_Part2, XL_Part3)
    do i=1,nx
        x1d(i)=x1d(i)+x_begin
    enddo

    open(99,file="xx.dat")
    do i=1,nx-1
        write(99,*)  i, x1d(i), x1d(i+1)-x1d(i)
    enddo
    close(99)
end

! 周向网格： 均匀
subroutine comput_phi0(ns,Phi0, If_half )
    implicit none
    integer:: ns, j, If_half
    real*8:: phi0(ns), SL
    real*8,parameter::  PI=3.1415926535897932d0
    if(If_half .eq. 1)  then
        SL=PI
    else
        SL=2.d0*PI
    endif

    do j=1,ns
        Phi0(j)=SL*(j-1.d0)/(ns-1.d0)
    enddo
end

!-------------------------------------------------------
subroutine wallmesh2d(nx,ns,x1d,Phi0,x2d,y2d,z2d,If_plotwallmesh)
    implicit none
    integer:: nx,ns,i,j,k,If_plotwallmesh
    real*8:: x2d(nx,ns),y2d(nx,ns),z2d(nx,ns),x1d(nx),Phi0(ns),x,y,z,phi

    do i=1,nx
        x=x1d(i)
        do k=1,ns
            phi= Phi0(k)
            call wallmesh(phi,x,y,z)
            x2d(i,k)=x
            y2d(i,k)=y
            z2d(i,k)=z
        enddo
    enddo

    If( If_plotwallmesh .eq. 1) then
        print*, "write 2D wall mesh"
        open(99,file="wallmesh2d.dat")
        write(99,*) "variables= x,y,z"
        write(99,*) "zone i=",nx, " j= ", ns
        do j=1,ns
            do i=1,nx
                write(99,"(3F16.8)") x2d(i,j), y2d(i,j), z2d(i,j)
            enddo
        enddo
        close(99)
    endif

end

 !-----------------------------------------------------------
subroutine Wall_Normal(nx,ns,x2d,y2d,z2d,xn,yn,zn,Phi0)
    implicit none
    integer:: nx,ns,i,j
    real*8,dimension(nx,ns):: x2d,y2d,z2d,xn,yn,zn
    real*8:: Phi0(ns), Phi, sn , xn1, yn1, zn1, xn2,yn2,zn2, dx, ds
    real*8:: x1,y1,z1, x2,y2,z2

    dx=1.d-3
    ds=1.d-3

    do j=1,ns
        phi=Phi0(j)
        do i=1,nx
            call wallmesh(phi, x2d(i,j)-dx,  y1,  z1)
            call wallmesh(phi, x2d(i,j)+dx,  y2,  z2)
            xn1=1.d0;   yn1=(y2-y1)/(2.d0*dx) ;  zn1 = (z2-z1)/(2.d0*dx)
            call wallmesh(phi-ds, x2d(i,j), y1, z1)
            call wallmesh(phi+ds, x2d(i,j), y2, z2)
            xn2=0.d0 ;  yn2= (y2-y1)/(2.d0*ds) ;  zn2=(z2-z1)/(2.d0*ds)
            xn(i,j)=yn1*zn2-yn2*zn1
            yn(i,j)= - (xn1*zn2-xn2*zn1)
            zn(i,j)=xn1*yn2-xn2*yn1
            sn=1.d0/sqrt(xn(i,j)**2+yn(i,j)**2+zn(i,j)**2)
            xn(i,j)=xn(i,j)*sn;  yn(i,j)=yn(i,j)*sn ;  zn(i,j)= zn(i,j)*sn                             ! 法方向
        enddo
    enddo
end

!--------------- input phi, x ;  output y, z
subroutine  wallmesh(phi, x,y,z)
    implicit none
    integer:: k,i
    real*8:: x,y,z,phi,Rd,x0,beta,csg,sng,xe,Re,R,sk
    real*8,parameter:: Lx=1600.d0, a1=20.d0, b1=40.d0, c1=20.d0
    real*8,parameter:: PI=3.1415926535897932d0

    call comput_Rd(Phi, Rd)          ! 计算底面半径R
    beta=1.d0/sqrt(  (cos(Phi)/c1)**2 + (sin(Phi)/b1)**2 )
    x0=Lx-a1
    csg=(a1*a1*beta*Rd+ beta*x0*sqrt((Rd*a1)**2+(beta*x0)**2 - (beta*a1)**2 )  ) / (Rd*Rd*a1*a1+beta*beta*x0*x0)
    sng=sqrt(1.d0-csg*csg)
    xe=a1-a1*sng
    Re=beta*csg
    sk=(Rd-Re)/(x0-a1*sng)
    if(x<= xe) then
        R=sqrt(1.d0- ((x-a1)/a1)**2) *beta
    else
        R=Re+sk*(x-xe)
    endif
    y=R*sin(Phi)
    z=R*cos(Phi)

end

 !--------------- input phi-----------------------------------------
subroutine  comput_xe_sk(phi, xe,Re,sk)                ! (xe,Re): 连接点坐标；  sk 母线斜率
    implicit none
    integer:: k,i
    real*8:: x,y,z,phi,Rd,x0,beta,csg,sng,xe,Re,R,sk
    real*8,parameter:: Lx=1600.d0, a1=20.d0, b1=40.d0, c1=20.d0
    real*8,parameter:: PI=3.1415926535897932d0

    call comput_Rd(Phi, Rd)          ! 计算底面半径R
    beta=1.d0/sqrt(  (cos(Phi)/c1)**2 + (sin(Phi)/b1)**2 )
    x0=Lx-a1
    csg=(a1*a1*beta*Rd+ beta*x0*sqrt((Rd*a1)**2+(beta*x0)**2 - (beta*a1)**2 )  ) / (Rd*Rd*a1*a1+beta*beta*x0*x0)
    sng=sqrt(1.d0-csg*csg)
    xe=a1-a1*sng
    Re=beta*csg
    sk=(Rd-Re)/(x0-a1*sng)
end


!-------------------给定子午角 Phi,  计算底面的半径 R=sqrt(y*y+z*z) ---------------------
subroutine comput_Rd(Phi, R)
    implicit none
    real*8:: Phi, R, Phi1
    real*8,parameter:: PI=3.14159265358979324d0
    if(Phi <0 )  Phi=Phi+2.d0*PI
    if(Phi > 2.d0*PI)  Phi=Phi-2.d0*PI

    if(Phi .ge. 0.5*PI   .and. Phi .le. 1.5*PI) then
        call comput_R_E1(Phi,R)
    else
        call comput_R_E2(Phi, R)
    endif
end


! in the Ellipic
subroutine comput_R_E1(Phi,R)
    implicit none
    real*8,parameter::    b2=300.d0, c2=75.d0
    real*8:: Phi, R
    R=sqrt(1.d0/( (sin(Phi)/b2)**2+ (cos(Phi)/c2)**2  ) )
end

! in the neighber of Ellipic
subroutine comput_R_E2(Phi,R)
    implicit none
    real*8,parameter::   Yx=600.d0,  Zx=270.d0,  b2=300.d0, c2=75.d0,  b3=300.d0, c3=Zx
    real*8,parameter:: epsl=1.d-8
    real*8:: Phi,R, Rnew,alfa,zc,y,z,eta,dR
    alfa=1.d0-c2/Zx
    R=270.d0
    do
        y=R*sin(Phi)
        z=R*cos(Phi)
        eta=y/Yx+0.5d0
        zc=Zx*256.d0*eta**4*(1.d0-eta)**4            ! CST curve
        Rnew=sqrt( (1.d0 +(2.d0*alfa*z*zc-(alfa*zc)**2)/c2**2 )     / ( ( sin(Phi)/b2)**2+ (cos(Phi)/c2)**2 )    )
        dR=abs(Rnew-R)
        R=Rnew
        if(dR<epsl) exit
    enddo
end


!   产生一维等比拉伸（指数）网格。 给定总长度SL, 第1个网格间距deltx0和网格数目nx, 产生等比网格序列xx(:)
subroutine grid1d_exponential(nx,xx,Length,deltx0)
    implicit none
    integer::nx,i
    real*8 xx(nx), Length, deltx0, dx,delta,fb,fbx,bnew,a,s
    real*8,save:: b=3.d0
    dx=1.d0/(nx-1.d0)
!---------------------------------------
    delta=deltx0/Length
! using Newton method get coefficient
100 continue
    fb=(exp(b/(nx-1.d0))-1.d0)/(exp(b)-1.d0)-delta
    fbx=(exp(b/(nx-1.d0))/(nx-1.d0)*(exp(b)-1.d0) -      &
        (exp(b/(nx-1.d0))-1.d0)*exp(b)  )/((exp(b)-1.d0))**2
    bnew=b-fb/fbx
    if(abs(b-bnew) .gt. 1.d-6) then
        b=bnew
        goto 100
    endif
    b=bnew
    a=1.d0/(exp(b)-1.d0)
    do i=1,nx
        s=(i-1.d0)*dx
        xx(i)=a*(exp(s*b)-1.d0)*Length
    enddo
end



!   产生一维等比拉伸（指数）网格。 给定总长度SL, 第1个网格间距deltx0和网格数目nx, 产生等比网格系数expa,expb
subroutine grid1d_exponential2(nx,Length,deltx0,expa,expb)
    implicit none
    integer::nx,i
!	     real*8 xx(nx), Length, deltx0, dx,delta,fb,fbx,bnew,a,s
    real*8  Length, deltx0, dx,delta,fb,fbx,bnew,a,expa,expb
    real*8,save:: b=3.d0
    dx=1.d0/(nx-1.d0)
!---------------------------------------
    delta=deltx0/Length
! using Newton method get coefficient
100 continue
    fb=(exp(b/(nx-1.d0))-1.d0)/(exp(b)-1.d0)-delta
    fbx=(exp(b/(nx-1.d0))/(nx-1.d0)*(exp(b)-1.d0) -      &
        (exp(b/(nx-1.d0))-1.d0)*exp(b)  )/((exp(b)-1.d0))**2
    bnew=b-fb/fbx
    if(abs(b-bnew) .gt. 1.d-6) then
        b=bnew
        goto 100
    endif
    b=bnew
    a=1.d0/(exp(b)-1.d0)
    expa=a
    expb=b
end


!-------- 四段网格分布--------------
! -- 均匀网格 --过渡网格--均匀网格--等比拉伸网格----

subroutine get_sx_4part(nx, XL, xx, nx_part1,nx_part3, XL_part1, XL_Part2, XL_Part3)
    implicit none
    integer::nx,nx_part1, nx_part2,nx_part3, nx_buff
    integer:: i
    real*8:: xx(nx),XL, XL_part1, XL_Part2, XL_part3, dx1, dx3, XL_buff

    dx1=XL_part1/nx_part1
    dx3=XL_part3/nx_part3
    nx_part2=int(3.d0*XL_part2/(2.d0*dx1+dx3)+1.d0 +0.5d0)
    nx_buff=nx-nx_part1-nx_part2-nx_part3
    XL_buff=XL-XL_part1-XL_part2-XL_part3

    if(nx_buff <0 )  then
        print*, "warning !  nx_buff<0 !"
        stop
    endif

    print*,  "nx_part1, nx_Part2, nx_part3, nx_buff=",  nx_part1, nx_Part2, nx_part3, nx_buff


!------三段网格；  Part1, Part3 均匀; Part2 过渡
    do i=1,nx_part1
        xx(i)=(i-1.d0)*dx1
    enddo

    do i=nx_part1+nx_part2+1,nx_part1+nx_part2+nx_part3
        xx(i)=XL_part1+XL_part2+(i-nx_part1-nx_part2)*dx3
    enddo

    call grid1d_transition(nx_part2,xx(nx_part1+1), XL_part2, dx1,dx3)

    do i=nx_part1+1,nx_part1+nx_part2
        xx(i)=xx(i)+XL_part1
    enddo


    call grid1d_exponential(nx_buff,xx(nx-nx_buff+1),XL_buff-dx3, dx3)


    do i=nx_part1+nx_part2+nx_part3+1, nx
        xx(i)=xx(i)+XL_part1+XL_part2+XL_part3+dx3
    enddo

end

!--------------------------------------------------------------
! 模块2: 过渡网格，采用三次函数过渡，光滑连接两端
! 接口：nx 网格数； Length 长度； dx1 首端网格间距； dx2 尾端网格间距
! 通常要求 平均网格间距 Length/(nx-1.) 介于 dx1 与dx2之间
!--------------------------------------------------------------

subroutine grid1d_transition(nx,xx, Length, dx1,dx2)
    implicit none
    integer:: nx,i
    real*8 Length, xx(nx), dx1, dx2, dx0,x,Ah1,Sh0,Sh1
    dx0= Length/(nx-1.d0)
    if( dx0 .gt. dmax1(dx1,dx2) .or. dx0 .lt. dmin1(dx1,dx2) ) then
        print*, "warning !  dx0 should between dx1 and dx2 !!!"
        print*, "dx0  dx1 and dx2 =", dx0, dx1, dx2
    endif

    do i=1,nx
        x=(i-1.d0)/(nx-1.d0)
!     Ah0=(1.d0+2.d0*x)*(x-1.d0)**2    ! 三次Hermit插值的4个基函数
        Ah1=(1.d0-2.d0*(x-1.d0))*x*x
        Sh0=x*(x-1.d0)**2
        Sh1=(x-1.d0)*x*x
        xx(i)=Length*Ah1+dx1*(nx-1.d0)*Sh0+dx2*(nx-1.d0)*Sh1
    enddo
end


subroutine comput_zh(nx,ns,zh,Phi0,h_begin,h_end1,h_end2,h_end_seta0,IF_half)
    implicit none
    integer:: nx,ns,i,j,IF_half
    real*8:: zh(nx,ns),Phi0(ns),zh0(ns),h_begin,h_end1,h_end2,h_end_seta0, x
    real*8,parameter::  PI=3.1415926535897932d0
    if(IF_half .eq. 0)   then
        print*, "Warning ! In this version, If_half should be in  @ in comuput_zh ! "
        stop
    endif

    do i=1,ns
        if(Phi0(i) <= h_end_seta0)  then
            zh0(i)=h_end1
        else
            x=(Phi0(i)- h_end_seta0)/(Pi-h_end_seta0)
            zh0(i)= h_end1*(1.d0+2.d0*x)*(x-1.d0)**2  + h_end2*(1.d0-2.d0*(x-1.d0))*x*x             ! 三次Hermint 插值（两端导数为0）
        endif
    enddo

    do j=1,ns
        do i=1,nx
            zh(i,j)=h_begin+(i-1.d0)*(zh0(j)-h_begin)/(nx-1.d0)                 ! 不包含头部，均匀网格
        enddo
    enddo

    open(99,file="zh_end.dat")
    do j=1,ns
        write(99,*) Phi0(j), zh0(j)
    enddo
    close(99)
end

