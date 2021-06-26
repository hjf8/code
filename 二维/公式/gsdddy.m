function X=gsdddy(A,b,X0,P,wucha,max1)
D=diag(diag(A));U=-triu(A,1);
L=-tril(A,-1);dD=det(D);
if dD==0
    disp('对角矩阵D奇异，无解.')
else
    disp('对角矩阵D非奇异，有解.')
    iD=inv(D-L);B2=iD*U;f2=iD*b;jX=A\b;
    X=X0;[n m]=size(A);
    for k=1:max1
        X1=B2*X+f2;djwcX=norm(X1-X,P);
        xdwcX=djwcX/(norm(X,P)+eps);
        if(djwcX<wucha)|(xdwcX<wucha)
            return
        else
            k,X1',k=k+1;X=X1;
        end
    end
    if(djwcX<wucha)|(xdwcX<wucha)
        disp('请注意：高斯—赛德尔迭代收敛，此A的分解矩阵D,U,L和精确解jX和近似解X如下：')
    else
        disp('迭代结果没有达到给定的精度，迭代次数已超过最大迭代次数max1,精确解jX和迭代向量X如下：')
        X=X';jX=jX'
    end
end
X=X';D,U,L,jX=jX'