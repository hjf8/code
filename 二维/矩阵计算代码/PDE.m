function u=PDE(n,J)
%n表示时间层，J表示空间层
tic;
J=100;n=3200;
h=1/J;%空间步长
H=0:h:1;
ta=1/n;
r=ta/(h^2);
A=zeros(J+1);
A(1,1)=1+2*r;A(1,2)=-2*r;
A(J+1,J)=-2*r;A(J+1,J+1)=1+2*r;
for i=2:J
    A(i,i)=1+2*r;
    A(i,i-1)=-r;
    A(i,i+1)=-r;
end

% right hand vector
u0=zeros(J+1,1);
for i=1:J+1
    u0(i)=cos(pi*(i-1)*h);
end
U=zeros(J+1,n+1);
U(:,1)=u0;%每一列代表一次时间上的迭代解
b=zeros(J+1,1);
for i=1:n
    for j=1:J+1
        b(j)=u0(j)+ta*sin(i*ta);
    end
    u1=inv(A)*b; % 直接法
    %u1=LU(A,b,2,2); % LU分解
    %[u1]=GivensQR(A,b); % QR分解
    %wucha=10e-10;N=1000;x0=ones(J+1,1); %N表示迭代次数，x0表示初始值
    %[u1]=Jacobi(A,b,x0,wucha,N); % 雅可比迭代法
    %[u1]=Gsddy(A,b,x0,wucha,N);
    %[u1]=gradient(A,b,x0,wucha,N);%梯度法
    u0=u1;
    U(:,i+1)=u0;
end
U;
un=U(:,n+1);
subplot(2,1,1);
scatter(H,u0);
hold on
% true solution
x=0:h:1;
y= exp(-pi*pi)*cos(pi*x)+1-cos(1);
plot(x,y)
xlabel('x');ylabel('u(the last time layer)');
legend('real solution','numerical solution');
title('comparison of the solution')
subplot(2,1,2);
e=zeros(J+1,1);
for i=1:J+1
    e(i)=single(e(i));
    e(i)=abs(y(i)-un(i));
end
%m=max(e)
scatter(H,e);
legend('error');
title('the error')
t1=toc


