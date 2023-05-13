import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Application {
    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("spark-mllib").master("local").getOrCreate();

        Dataset<Row> raw_data = sparkSession.read().format("csv")
                .option("header","true")
                .option("interSchema", "true")
                .load("C:\\Users\\hazal\\Desktop\\satis.csv");

        VectorAssembler features_vector = new VectorAssembler().setInputCols(new String[] {"Ay"})
                .setOutputCol("features");

        Dataset<Row> transform = features_vector.transform(raw_data);

        Dataset<Row> final_data = transform.select("features", "Satis");

        Dataset<Row>[] datasets = final_data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> train_data = datasets[0];
        Dataset<Row> test_data = datasets[1];

        LinearRegression lr = new LinearRegression();
        lr.setLabelCol("Satis");

        LinearRegressionModel model = lr.fit(train_data);

        Dataset<Row> transform_test = model.transform(test_data);

        transform_test.show();

    }
}
