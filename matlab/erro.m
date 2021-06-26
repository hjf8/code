function [E B]=erro(a,b,n,m,lambda)%E为误差，B为误差界
a=0;b=1;n=2;lambda=5;m=5;
syms t u
x0=exp(-u)*cos(u);%真解,注意！自变量为u
y=const_y(lambda,a,b);y=subs(y,t,u);
c=middle_c(m,n,a,b,lambda);
h=(b-a)/m;%对t的剖分
for i=1:m+1
    t(i)=a+(i-1)*h;
end
x0=subs(x0,u,t');x0=vpa(x0,10);
y=subs(y,u,t');y=vpa(y,10);
alafa=[];
 for j=1:n
     alafa(:,j)=t'.^(j-1)/factorial(j-1);%m+1乘n矩阵，alafa在插值点值得矩阵
 end
 E=ones(n,1);
 for j=1:n
     d1=ones(n,1);d1(j+1:n)=0;d2=d1;
     x=1/lambda*(y+alafa*(c.*d2));
     E(j)=norm(x-x0,inf);
     E=vpa(E',10);
 end
E
%end

