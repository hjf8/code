function [L U]=BlockLU(A,n,r)
if n<=r
    for j=1:n-1
        A(j+1:n,j)=A(j+1:n,j)/A(j,j);
        A(j+1:n,j+1:n)=A(j+1:n,j+1:n)-A(j+1:n,j)*A(j,j+1:n);
    end
    L=eye(n)+tril(A,-1),
    U=triu(A),
else
    for k=1:r
        p=k+1:n;
        A(p,k)=A(p,k)/A(k,k);
        if k<r
            u=k+1:r;
            A(p,u)=A(p,u)-A(p,k)*A(k,u);
        end
    end
    L11=eye(r)+tril(A(1:r,1:r),-1),U11=triu(A(1:r,1:r)),
    U12=L11\A(1:r,r+1:n);L21=A(r+1:n,1:r);
    H=A(r+1:n,r+1:n)-L21*U12;
    [L U]=BlockLU(H,n-r,r);
   % L=[L11,zeros(r,n-r);L21,L22];
    %U=[U11,U12;zeros(n-r,r),U22];
end

    