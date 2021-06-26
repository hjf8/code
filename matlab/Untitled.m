max=100;
v=ones(1,9);
B=2*eye(10);
C=diag(v,1);
D=diag(v,-1);
A=B+C+D;
X0=ones(1,10)';
b=zeros(10,1);
b(1,1)=1;b(10,1)=1;
D1=diag(diag(A));dD=det(D1);
if dD==0
    disp('对角矩阵D奇异，无解.')
else
    disp('对角矩阵D非奇异，有解.')
    iD=inv(D1);B2=iD;jX=A\b;%jx是精确解
    X=X0;
    for k=1:max
        X1=X+B2*(b-A*X);djwcX=norm(X1-X,Inf);
        if(djwcX<wucha)
            return
        else
            k,X1',k=k+1;X=X1;
        end
    end
    if(djwcX<wucha)|(xdwcX<wucha)
        disp('请注意：高斯—赛德尔迭代收敛，此A的分解矩阵D,U,L和精确解jX和近似解X如下：')
    else
        disp('迭代结果没有达到给定的精度，迭代次数已超过最大迭代次数max,精确解jX和迭代向量X如下：')
        X=X';jX=jX'
    end
end
X=X';D,U,L,jX=jX'
t=[];
n=10;
for k=1:n
    for j=1:10
        x(j)=(b(j)-A(j,[1:j-1,j+1:10])*x0([1:j-1,j+1:10]))/A(j,j);
    end
    t=[t,x];
    x0=x';
end
disp('方程精确解:');
x1
disp('方程迭代10步后的解:');
x
disp('方程每次迭代的解:');
m=reshape(t',10,10)
save data3 A b m x1