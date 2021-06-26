function u=LU(A,b,p,q)
%q表示上带宽，p表示下带宽
n=size(A);
%带状矩阵的LU分解
for k=1:n-1
    for i=k+1:min(k+p,n)
        A(i,k)=A(i,k)/A(k,k);
    end
    for j=k+1:min(k+q,n)
        for i=k+1:min(k+p,n)
            A(i,j)=A(i,j)-A(i,k)*A(k,j);
        end
    end
end
L=eye(n)+tril(A,-1);
U=triu(A);
%带状三角方程组求解
%向前消去法：列形式
for j=1:n
    for i=j+1:min(j+p,n)
        b(i)=b(i)-L(i,j)*b(j);
    end
end
%向后消去法;行形式
for j=n:-1:1
    b(j)=b(j)/U(j,j);
    for i=max(1,j-p):j-1
        b(i)=b(i)-U(i,j)*b(j);
    end
end
u=b;