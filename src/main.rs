use std::error::Error;
use std::fs::File;
use std::path::Path;
use csv::Writer;
use glf::GLF;

fn main() -> Result<(), Box<dyn Error>> {
    let path = Path::new("/Users/eirikvarnes/Desktop/test_data/06.08 pipeline_medium_test/8_run_short_range_2x_metal_pipe_rocks/log_2024-08-06-101206.glf");
    if !path.exists() {
        eprintln!("GLF file not found at {:?}", path);
        return Ok(());
    }

    let glf = GLF::new(path).unwrap();
    
    let mut wtr = Writer::from_writer(File::create("image_records.csv")?);

    // Write the header for the CSV
    wtr.write_record(&["Header", "Version", "Image Version", "Range Start", "Range End", "Range Compression",
                       "Bearing Start", "Bearing End", "Compression Type", "Data Pointer", "Data Size", 
                       "Bearing Table", "State Flags", "Modulation Frequency", "Beam Form App", "Transmission Time",
                       "Ping Flags", "SOS at XD", "Percentage Gain", "Chirp Mode", "Sonar Type", "Platform ID", 
                       "Record Size", "Image Width", "Image Height"])?;

    for (_index, image) in glf.images.iter().enumerate() {
        wtr.write_record(&[
            format!("{:?}", image.header),
            image.version.to_string(),
            image.image_version.to_string(),
            image.range_start.to_string(),
            image.range_end.to_string(),
            image.range_compression.to_string(),
            image.bearing_start.to_string(),
            image.bearing_end.to_string(),
            image.compression_type.to_string(),
            image.data_ptr.to_string(),
            image.data_size.to_string(),
            format!("{:?}", image.bearing_table),
            image.state_flags.to_string(),
            image.modulation_frequency.to_string(),
            image.beam_form_app.to_string(),
            image.db_tx_time.to_string(),
            image.ping_flags.to_string(),
            image.sos_at_xd.to_string(),
            image.percent_gain.to_string(),
            image.chirp.to_string(),
            image.sonar_type.to_string(),
            image.platform.to_string(),
            image.record_size.to_string(),
            image.image_width.to_string(),
            image.image_height.to_string()
        ])?;
    }

    wtr.flush()?;

    let mut status_wtr = Writer::from_writer(File::create("status_records.csv")?);

    // Write the header for the CSV
    status_wtr.write_record(&["Header", "BF Version", "DA Version", "Flags", "Device ID", "XD Selected", "VGA T1",
                              "VGA T2", "VGA T3", "VGA T4", "PSU Temperature", "Die Temperature", "Transmit Temperature",
                              "AFE0 Top Temperature", "AFE0 Bottom Temperature", "AFE1 Top Temperature", 
                              "AFE1 Bottom Temperature", "AFE2 Top Temperature", "AFE2 Bottom Temperature", 
                              "AFE3 Top Temperature", "AFE3 Bottom Temperature", "Link Type", "Uplink Speed",
                              "Downlink Speed", "Link Quality", "Packet Count", "Receive Error Count", "Resent Packet Count",
                              "Dropped Packet Count", "Unknown Packet Count", "Lost Line Count", "General Count", 
                              "Sonar Alt IP", "Surface IP", "Subnet Mask", "MAC Address", "Boot Status Register", 
                              "Boot Status Register DA", "FPGA Time", "Dip Switch", "Shutdown Status", "Net Adapter Found"])?;

    for status in &glf.statuses {
        status_wtr.write_record(&[
            format!("{:?}", status.header),
            status.bf_version.to_string(),
            status.da_version.to_string(),
            status.flags.to_string(),
            status.device_id.to_string(),
            status.xd_selected.to_string(),
            status.vga_t1.to_string(),
            status.vga_t2.to_string(),
            status.vga_t3.to_string(),
            status.vga_t4.to_string(),
            status.psu_t.to_string(),
            status.die_t.to_string(),
            status.tx_t.to_string(),
            status.afe0_top_temp.to_string(),
            status.afe0_bot_temp.to_string(),
            status.afe1_top_temp.to_string(),
            status.afe1_bot_temp.to_string(),
            status.afe2_top_temp.to_string(),
            status.afe2_bot_temp.to_string(),
            status.afe3_top_temp.to_string(),
            status.afe3_bot_temp.to_string(),
            status.link_type.to_string(),
            status.uplink_speed.to_string(),
            status.downlink_speed.to_string(),
            status.link_quality.to_string(),
            status.packet_count.to_string(),
            status.recv_error.to_string(),
            status.resent_packet_count.to_string(),
            status.dropped_packet_count.to_string(),
            status.unknown_packet_count.to_string(),
            status.lost_line_count.to_string(),
            status.general_count.to_string(),
            status.sonar_alt_ip.to_string(),
            status.surface_ip.to_string(),
            format!("{:?}", status.subnet_mask),
            format!("{:?}", status.mac_addr),
            status.boot_sts_register.to_string(),
            status.boot_sts_register_da.to_string(),
            status.fpga_time.to_string(),
            status.dip_switch.to_string(),
            status.shutdown_status.to_string(),
            status.net_adap_found.to_string()
        ])?;
    }

    status_wtr.flush()?;

    // Example of extracting and saving an image
    if let Some(img) = glf.extract_image(1573).ok() {
        img.save("test.png").unwrap();
        println!("Extracted image 1 and saved as test.png");
    }

    Ok(())
}
