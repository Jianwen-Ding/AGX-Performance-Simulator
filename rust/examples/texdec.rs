fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    for hx in a {
        let bytes = (0..hx.len()).step_by(2).map(|i| u8::from_str_radix(&hx[i..i + 2], 16).unwrap()).collect::<Vec<_>>();
        let mut full = vec![0u8; 16];
        full[..bytes.len()].copy_from_slice(&bytes);
        let n0 = agx_sim::isa::opcode_to_number(&full);
        let desc = agx_sim::isa::get_instruction_descriptor(n0).unwrap();
        let size = desc.decode_size(n0);
        let n = desc.mask_instr(n0);
        let fields = desc.decode_fields(n);
        let ops = desc.fields_to_operands(&fields).unwrap();
        println!("{} -> {} ({}B)", hx, desc.name, size);
        let mut fs: Vec<(&str, u64)> = fields.iter().map(|(k, v)| (*k, *v)).collect();
        fs.sort();
        for (k, v) in fs {
            println!("    {:>4} = {:#x} ({})", k, v, v);
        }
        for o in &ops {
            println!("    op {:?}", o);
        }
    }
}
