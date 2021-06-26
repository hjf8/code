function [x,m]=CG(A,b,x0,tol,times)
n=size(A,1);%n=����A������
x=x0;
r=b-A*x;%����в�r�ĳ�ʼֵ
error = norm(r)
p=r;%p(k)=r(k)�����
for k=1:times
  alpha=(r'*r)/(p'*A*p);%alpha��k���Ķ���
  x=x+alpha*p;%x(k+1)=x(k)+alpha(k)*p(k)
  r2=r-alpha*A*p;%r(k+1)=r(k)-alpha*A*p
  error = norm(r2)%r(k+1)�ķ���
  m=k%��������
  if (error<=tol)
    break;
  end
  beta=(r2'*r2)/(r'*r);
  p=r2+beta*p;
  r=r2;
end
end
