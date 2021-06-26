function X=gsd(A,b,X0,wucha,max1)
%wucha=10e-10;max1=100;
%v=ones(1,2);
%B=[1 2 1];
%A=diag(B)-diag(v,1)-diag(v,-1);
%X0=[0;0;0];
%b=[-1;-1;2];
D=diag(diag(A));L=tril(A,-1);
iD=inv(D+L);B2=iD;
X=X0;
for k=1:max1
    X1=X+B2*(b-A*X);
    djwcX=norm(X1-X,Inf);
    if(djwcX<wucha)
        disp('高斯—赛德尔迭代收敛，迭代次数k与近似解X1如下：')
        k,X1
        return
    else
        k=k+1;X=X1;
        if (k>max1)
            disp('迭代结果没有达到给定的精度，迭代次数已超过最大迭代次数max1,迭代向量X1如下：')
            X1
            return
        end
    end
end