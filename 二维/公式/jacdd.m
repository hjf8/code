function X=jacdd(A,b,X0,P,wucha,max1)
[n,m]=size(A);
for j=1:m
    a(j)=sum(abs(A(:,j)))-2*(abs(A(j,j)));
end
for i=1:n
    if a(i)>=0
        disp('请注意，A不是严格对角占优，此雅可比迭代不一定收敛')
        return
    end
end
if a(i)<0
    disp('请注意：A严格对角占优，此方程组有唯一解且雅可比迭代收敛')
end
for k=1:max1
    k
    for j=1:m
        X(j)=(b(j)-A(j,[1:j-1,j+1:m])*X0([1:j-1,j+1:m]))/A(j,j);
    end
    X,djwcX=norm(X'-X0,P);xdwcX=djwcX/(norm(X',P)+eps);X0=X';X1=A\b;
    if (djwcX>wucha)&(xdwcX<wucha)
        disp('请注意：雅可比迭代收敛，此方程组的精确解jX和近似解X如下：')
        return
    end
end
if (djwcX>wucha)&(xdwcX>wucha)
    disp('请注意：雅可比迭代次数已经超过最大迭代次数max1')
end
a,X=X;jX=X1',