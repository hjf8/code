function [u x]=new(a,b,n,u)
%a=0;b=2*pi;n=100;u=zeros(100);
h=(b-a)/n;
x=[a:h:b];
A(1:n+1,1:n+1)=0;
A(1,1) =  1.0;
A(n+1,n+1) =  1.0;
R(1,1) = 0.0;
R(n+1,1)=0.0;
for i=2:n
   R(i,1)=(sin(x(i))+(sin(x(i)))^3-2*(u(i))^3)*h^2;
   A(i,i)=2+3*h^2*diag(u(i));
   A(i,i-1)=-1;
   A(i,i+1)=-1;
end
u = A\R;
x;
%plot(x,u)