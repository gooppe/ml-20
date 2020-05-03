# Подсчет числа слов в документе

### Сборка
```bash
sbt package
```

### Запуск
```bash
spark-submit --class "WordCounter" target/scala-2.11/word-counter_2.11-1.0.jar input.txt output
cat output/part-* > output.csv
```

### Пример работы
```bash
$ head -20 input.txt
> Richard Bach. Jonathan Livingston Seagull
> 
> 
>                                              To the real Jonathan Seagull,
>                                              who lives within us all.
> 
> 
> Part One
> 
> 
>      It was morning, and the new sun sparkled gold across the ripples of a
> gentle sea. A mile from shore a fishing boat chummed the  water.  and  the
> word for Breakfast Flock flashed through  the  air,  till  a  crowd  of  a
> thousand seagulls came to dodge and fight for bits of food. It was another

$ spark-submit --class "WordCounter" target/scala-2.11/word-counter_2.11-1.0.jar input.txt output
$ cat output/part-* > output.csv
$ head -7 output.csv
> miss,1
> mattered,1
> someone,1
> ll,13
> bone,3
> Because,1
> shot,1
```

