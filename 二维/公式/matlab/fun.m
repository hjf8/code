function y=fun(a,b,n)
a=0;b=2*pi;n=100000;
h=(b-a)/n;
x=[a:h:b];
A=zeros(n+1);
A(1,1)=1;A(n+1,n+1)=1;
f=zeros(n+1,1);
f(1,1)=0;f(n+1,1)=0;
for i=2:n
    A(i,i)=2;
    A(i,i-1)=-1;
    A(i,i+1)=-1;
    f(i)=-sin(x(i))*h^2;
end
%u=A\f;
u=gs(A,f);
plot(x,u);
    
    