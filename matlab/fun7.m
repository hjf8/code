function [m]=fun7(n)
x=rand(n,1);
y=rand(1,n);
m=0;
k=1;
for i=1:n
    for j=1:n
        c(i,j)=[x(i)*y(j)].^k;
        m=m+1;
    end
end
disp(c);