function f=f(a,b,lamuda)
syms s u t v
u=(2*s-a-b)/(b-a);%s=0.5*[(b-a)*u+a+b]
v=(2*t-a-b)/(b-a);%t=0.5*[(b-a)*v+a+b]
u=[-0.774596662,0,0.774596662];
v=[-0.774596662,0,0.774596662];
A=[5/9,8/9,5/9];
for i=1:3
    f1=A(i)*(lamuda*sqrt(0.5*((b-a)*u(i)+a+b))*(0.5*((b-a)*u(i)+a+b))^(i-1));
end

for i=1:3
    for j=1:3
        f2=A(i)*A(j)*(exp(((b-a)^2*u(i)*v(j)+(b-a)*(b+a)*(u(i)+v(j))+(a+b)^2)/4)*sqrt(0.5*((b-a)*v(j)+a+b))*(0.5*((b-a)*u(i)+a+b))^(i-1));
    end
end

f=f1-f2;
