function [x,m]=CG(A,b,x0,tol,times)
n=size(A,1);%n=矩阵A的行数
x=x0;
r=b-A*x;%定义残差r的初始值
error = norm(r)
p=r;%p(k)=r(k)的情况
for k=1:times
  alpha=(r'*r)/(p'*A*p);%alpha（k）的定义
  x=x+alpha*p;%x(k+1)=x(k)+alpha(k)*p(k)
  r2=r-alpha*A*p;%r(k+1)=r(k)-alpha*A*p
  error = norm(r2)%r(k+1)的范数
  m=k%迭代次数
  if (error<=tol)
    break;
  end
  beta=(r2'*r2)/(r'*r);
  p=r2+beta*p;
  r=r2;
end
end
