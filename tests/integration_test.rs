use my_crate::hello;

#[test]
fn it_says_hello() {
    assert_eq!(hello(), "Hello, world!");
}
