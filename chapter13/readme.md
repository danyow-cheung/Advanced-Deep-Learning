# Unsupervised Learning using mutual information使用互信息進行無監督學習

許多機器學習任務，如分類、檢測和分割，都依賴於標記數據。 網絡在這些任務上的性能直接受到標記質量和數據量的影響。
問題在於生成足夠數量的高質量註釋數據既昂貴又耗時。

為了繼續推進機器學習的發展，新算法應該減少對人工標註者的依賴。 理想情況下，網絡應該從未標記的數據中學習
，由於互聯網的發展和智能手機和物聯網 (IoT) 等傳感設備的普及。 
從未標記的數據中學習是無監督學習的一個領域。 
在某些情況下，**無監督學習也被稱為自監督學習，強調使用純無標籤數據進行訓練**，無需人工監督。
在本文中，我們將使用術語無監督學習。


無監督學習的成功方法之一是最大化給定神經網絡中兩個隨機變量之間的互信息。 
在信息論領域，相互信息 (MI) 是兩個隨機變量之間依賴性的度量。

## Mutal information 相互信息
互信息是兩個隨機變量 X 和 Y 之間依賴性的度量。
有時，MI 也被定義為通過觀察 Y 獲得的有關 X 的信息量。
MI 也稱為信息增益或觀察 Y 時 X 的不確定性減少 .

由於 MI 可以揭示輸入、中間特徵、表示和輸出（它們本身是隨機變量）中依賴關係的重要方面，因此共享信息通常可以提高模型在下游任務中的性能。
一般來說，兩個隨機變量X,Y的關係有下面的定義。
$$
I(x;y)=D_k(P(x,y)||p(x)p(y))
$$
- P(x,y) 是 X 和 Y 在樣本空間 X x Y 上的聯合分佈。
- P(x)P(y) 是邊際分佈 P(X) 和 P(Y) 在樣本空間 X 和 Y 分別。

換句話說，MI 是聯合分佈與邊際分佈乘積之間的 Kullback-Leibler (KL) 散度


## Mutal information and Entropy 

$$
I(x;y)=D_k(P(x,y)||p(x)p(y)) = I(x;y) = H(x)+H(y)-H(x,y)
$$
上面意味著 MI 隨著邊際熵的增加而增加，但隨著聯合熵的增加而減少。 MI 在熵方面更常見的表達式如下：
$$
I(x;y)=D_k(P(x,y)||p(x)p(y)) = I(x;y) = H(x)- H(x|y)
$$
MI is how much decrease in information or uncertainty in X, had we known Y.
MI 是在我們知道 Y 的情況下，X 的信息或不確定性減少了多少。

$$
I(x;y)=D_k(P(x,y)||p(x)p(y)) = I(x;y) = H(y)- H(y|x)
$$

## Unsupervised learning by maxmizing the mutil Information of discrete random variables 
Our focus is on classification without labels. The idea is if we learn how
to cluster latent code vectors of all training data, then a linear separation algorithm can classify each test input data latent vector.
我們的重點是沒有標籤的分類。 這個想法是如果我們學習如何
對所有訓練數據的潛在代碼向量進行聚類，然後線性分離算法可以對每個測試輸入數據潛在向量進行分類。

To learn the clustering of latent code vectors without labels, our training objective is to maximize MI between the input image X and its latent code Y.
為了學習沒有標籤的潛在代碼向量的聚類，我們的訓練目標是最大化輸入圖像 X 及其潛在代碼 Y 之間的 MI。
mathematically,the objective is to maximize 
$$
I(x;y) = H(x)-H(x|y) 
$$

Invariant Information Clustering (IIC) proposed to meansure directly from joint and marginal distributions 
不變信息聚類 (IIC) 提議直接從聯合分佈和邊際分佈中求值

## Encoder network for unsupervised clustering 

## Unsupervised clustering implementation in Keras 
> unsupervised_clustering.py
the VGG backbone object is supplied during initializations. Given a backbone, the model is actually just a Dense layer with a softmax activation, as shown in the build_model() method. There is an option to create multiple heads.
VGG 骨幹對像在初始化期間提供。 給定主幹，模型實際上只是一個具有 softmax 激活的密集層，如 build_model() 方法所示。 可以選擇創建多個頭像。