function [x]=Gsddy(A,b,x0,wucha,N)
D1=diag(diag(A))+tril(A,-1);dD=det(D1);
%e=zeros(1,100);
if dD==0
    %disp('�ԽǾ���D���죬�޽�.')
else
    %disp('�ԽǾ���D�����죬�н�.')
    B=inv(D1);
    for k=1:N%��������
        x=x0+B*(b-A*x0);
        djwcx=norm(x-x0,Inf);%���
%        e(k)=djwcx;
        if(djwcx<wucha)%���Ƚ�
            break
        else
            k=k+1;x0=x;
        end
    end
    x0;
end
%plot(1:1:80,e(1:80))
%xlabel('��������')
%ylabel('error')
%title('Gsddy')