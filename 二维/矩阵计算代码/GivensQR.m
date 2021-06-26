function [u]=GivensQR(A,b)
n=size(A);
Q=eye(n);
for j=1:n-1
    B=eye(n);
    [c,s]=givens(A(j,j),A(j+1,j));
    B(j,j)=c;B(j+1,j+1)=c;B(j+1,j)=-s;B(j,j+1)=s;
    A(j:j+1,j:n)=[c,s;-s,c]'*A(j:j+1,j:n);
    Q=Q*B;
end
Q;R=A;
b=Q\b;
%向后消去法;行形式
for i=n:-1:1
    if(i<n)
        s=R(i,(i+1):n)*b((i+1):n,1);
    else
        s=0;
    end
    b(i)=(b(i)-s)/R(i,i);
end
u=b;