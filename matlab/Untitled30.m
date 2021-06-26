function [u]=GivensQR(A,b)
n=size(A);
Q=eye(n);
for j=1:n-1
    B=eye(n);
    [c,s]=givens1(A(j,j),A(j+1,j));
    B(j,j)=c;B(j+1,j+1)=c;B(j+1,j)=-s;B(j,j+1)=s;
    A(j:j+1,j:n)=[c,s;-s,c]'*A(j:j+1,j:n);
    Q=Q*B;
end
Q;R=A;
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