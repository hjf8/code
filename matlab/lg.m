function langrange=langrange(x,n)
langrange=0;
xxlinspace(-5,5,n+1);
for i=1:n+1
N=input('请输入插值节点数N=');
xx=-5:10/N:5;
ff=zeros(1,length(xx));
for i=1:(N+1)
    x=xx(i);
    ff(i)=eval(f);
end
M=-5:0.01:5;
output=zeros(1,length(M));
n=1;
for i=2:N+1
    for x=-5:0.01:5
        if x<xx(i)&&x>=xx(i-1)
            lx(1)=ff(i-1)*(x-xx(i))/(xx(i-1)-xx(i));
            lx(2)=ff(i)*(x-xx(i-1))/(xx(i)-xx(i-1));
            output(n)=lx(1)+lx(2);
            n=n+1;
        end
    end
end
ezplot(f,[-5,5])
hold on
A=-5:0.01:5;
plot(A,output,'r');