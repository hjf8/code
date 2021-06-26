function [x,m]=CG(A,b,x0,tol,times)
n=size(A,1);
x=x0;
r=b-A*x;
error = norm(r)
p=r;
for k=1:times
  alpha=(r'*r)/(p'*A*p);
  x=x+alpha*p;
  r2=r-alpha*A*p;
  error = norm(r2)
  m=k
  if (error<=tol)
    break;
  end
  beta=(r2'*r2)/(r'*r);
  p=r2+beta*p;
  r=r2;
end
end
