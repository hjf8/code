function budong(a,b,n)
a=0;b=2*pi;n=100;
h=(b-a)/n;
x=[a:h:b];
t=zeros(1,n);
u=0;
for i=1:n
    u=u-h^2*(-(u)^3+sin(x(i))+(sin(x(i)))^3);
    z(i)=u;
end
z
m=[1:100];
plot(m,z)

end