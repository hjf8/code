function [x]=Gsddy(A,b,x0,wucha,N)
D1=diag(diag(A))+tril(A,-1);dD=det(D1);
%e=zeros(1,100);
if dD==0
    %disp('对角矩阵D奇异，无解.')
else
    %disp('对角矩阵D非奇异，有解.')
    B=inv(D1);
    for k=1:N%迭代次数
        x=x0+B*(b-A*x0);
        djwcx=norm(x-x0,Inf);%误差
%        e(k)=djwcx;
        if(djwcx<wucha)%误差比较
            break
        else
            k=k+1;x0=x;
        end
    end
    x0;
end
%plot(1:1:80,e(1:80))
%xlabel('迭代次数')
%ylabel('error')
%title('Gsddy')