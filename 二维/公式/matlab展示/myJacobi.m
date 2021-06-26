function [x,n]=myJacobi(A,b,x0,tol,times)
D=diag(diag(A));
U=-triu(A,1);
L=-tril(A,-1);
iD=inv(D);
e=1;
n=1;
t=[];
B=iD*(L+U);
f=iD*b;
while e>tol
    t=[t,x0];
    x=B*x0+f;
    x0=x;
    e=norm(x-x0,inf);
    n=n+1;
end
    t;
%plot(n,e)