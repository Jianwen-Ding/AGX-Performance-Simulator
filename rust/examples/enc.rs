fn main() {
    let h = |b: Vec<u8>| b.iter().map(|x| format!("{:02x}", x)).collect::<String>();
    println!("{} {} {}", h(agx_sim::kernels::atomic_add_ret(13, 5, 21)), h(agx_sim::kernels::tg_atomic_add(13, 5, 21, false)), h(agx_sim::kernels::tg_atomic_add(13, 5, 21, true)));
    let p = agx_sim::cost::Params::new().unwrap();
    let ins = agx_sim::cost::decode(&agx_sim::kernels::tg_atomic_add(13, 5, 21, false), &p).unwrap();
    for i in ins { println!("{} {:?} {} dst {:?} src {:?}", i.name, i.kind, i.text, i.dst, i.src); }
}
