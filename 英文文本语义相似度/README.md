[比赛链接](https://god.yanxishe.com/53)  
在[Huggingface](https://github.com/huggingface/transformers#model-architectures)提供的transformer模型中二次开发,所以在运行本程序前需要运行`pip install transformers==2.7.0`
&nbsp;|model|折数|epochs|数据增强|result(%)
:--:|:--:|:--:|:--:|:--:|:--:|
1|roberta-base|0|30|无|85.67
2|roberta-base|5|10|无|83.87
3|roberta-base|5|30|无|85.81
4|roberta-large|5|10|无|89.87
5|roberta-base|5|10|有|88.42
6|roberta-large|5|10|有|90.17

数据增强是指自身数据进行增、删、别词替换、同义词替换等基于自身数据的增强操作，相当于扩大一定的数据量，引入一定的噪声，是模型的泛化性更强。
