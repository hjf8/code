a=0;b=1;n=10;
h=(b-a)/n;
s=[a:h:b];
A=zeros(n+1);
syms x
for i=1:n
    l=(x-s(i))/(-h)+1;
%    l=(x-s(i+1))*(x-s(i+2))/(2*h^2);
    m=(x-s(i+1))/h+1;
%    m=(x-s(i))*(x-s(i+2))/(-h^2);
    A(i,i+1)=A(i,i+1)+int(l*m,x,s(i),s(i+1));
    A(i+1,i)=A(i+1,i)+int(m*l,x,s(i),s(i+1));
    A(i,i)=A(i,i)+int(l*l,x,s(i),s(i+1));
    A(i+1,i+1)=A(i+1,i+1)+int(m*m,x,s(i),s(i+1));
end
A