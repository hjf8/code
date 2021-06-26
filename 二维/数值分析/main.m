function main(a,b,n)
a=0;b=2*pi;n=100;u=zeros(n+1,1);
for i=1:5
%    [u,x]=poisson(a,b,n,u);
    [u,x]=new(a,b,n,u);
    u;
    x;
end
u,x;
plot(x,u)