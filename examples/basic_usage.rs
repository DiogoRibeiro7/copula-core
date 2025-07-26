use copulas::{ClaytonCopula, Copula, VERSION};
fn main() -> Result<(), copulas::CopulaError> {
    let copula = ClaytonCopula::new(2.0)?;
    println!("Copulas version: {}", VERSION);
    println!("Copula dimension: {}", copula.dimension());
    Ok(())
}
