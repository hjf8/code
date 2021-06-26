function wangge(a,b,n)
a=0;b=1;n=10;
h=(b-a)/n;
A=[];
i=1; 
syms x
%for i=1
t(i)=a+i*h
l(i)=(x-t(i))./(-h)
l(i+1)=((x-a-(i-1)*h)/h)
l(i)*l(i+1)
A(i,i+1)=quad(@(x)(l(i)*l(i+1)),a-(i-1)*h,t(i));
A(i+1,i)=quad(l(i+1)*l(i),x,a-(i-1)*h,x(i));
A(i,i)=A(i,i)+int(l(i)*l(i),x,x(i-1),x(i));
A(i+1,i+1)=A(i+1,i+1)+int(l(i+1)*l(i+1),x,x(i-1),x(i));

    