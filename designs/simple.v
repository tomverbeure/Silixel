module counter(clock, out);

input clock;
output reg [7:0] out = 0;

always @(posedge clock)
begin
	out <= out + 1;
end

endmodule
