use copulas::archimedean::{AMHCopula, FrankCopula, JoeCopula};
use copulas::Copula;

#[test]
fn basic_construction() {
    let amh = AMHCopula::new(0.5).unwrap();
    assert_eq!(amh.dimension(), 2);

    let frank = FrankCopula::new(1.2).unwrap();
    assert_eq!(frank.dimension(), 2);

    let joe = JoeCopula::new(2.0).unwrap();
    assert_eq!(joe.dimension(), 2);
}
