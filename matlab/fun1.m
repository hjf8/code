function b=fun1(x,y)
c=x*y';
b=zeros(3);
%c=zeros(3);
%for i=1:3
%    for j=1:3
%        c(i,j)=c(i,j)+x(i)*y(j);
%    end
%end
for i=1:3
    for j=1:3
        for k=1:3
        b(i,j)=b(i,j)+c(i,k)*c(k,j);
        end
    end
end
disp(b);