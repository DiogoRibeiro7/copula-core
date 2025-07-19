use my_crate::hello;

fn main() {
    let greet = hello();
    println!("Custom: {}", greet.to_uppercase());
}
