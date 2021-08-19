mod model;
mod tensor;

#[cfg(test)]
mod tests {
    use crate::model::Model;

    #[test]
    fn it_works() {
        let model_path = "/Users/mustafaebraraktas/Projects/kiwi-rs/res/mfcc_calculator.tflite";
        let x = Model::new(model_path);

        assert_eq!(2 + 2, 4);
    }
}
