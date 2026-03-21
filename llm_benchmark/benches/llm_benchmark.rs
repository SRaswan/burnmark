#![feature(test)]
extern crate test;

#[cfg(test)]
mod benches {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_example(b: &mut Bencher) {
        b.iter(|| {
            let x = 2 + 2;
            test::black_box(x);
        });
    }
}
